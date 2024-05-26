from functools import lru_cache
import signal
import threading
import time
import traceback
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from jinja2 import Template

import torch
import uvicorn

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)
import asyncio
import json

from exllamav2.compat import safe_move_tensor
from exllamav2.generator.dynamic import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob
from exllamav2.generator.sampler import ExLlamaV2Sampler
from exllamav2.model import _torch_device

from illama.util import (
    Chat,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionsRequest,
    ChatStatus
)


class ILlamaServer:


    def __init__(self, ip: str, port: int, model_path: str, batch_size: int = 2, verbose: bool = False):
        self.ip: str = ip
        self.port: int = port
        self.model_path: str = model_path
        self.batch_size: int = batch_size
        self.verbose: bool = verbose
        
        self.model: ExLlamaV2 = None
        self.tokenizer: ExLlamaV2Tokenizer = None
        self.cache: ExLlamaV2Cache = None

        self.chat_template: str = None
        self.chats_lock = asyncio.Lock()
        self.chat_queue: asyncio.Queue[Chat] = asyncio.Queue()
        self.active_chats: Dict[int, Chat] = {}
        self.chat_template = None
        self.abort_event = threading.Event()
        
        
    def add_routes(self):
        self.app.add_api_route(
            "/v1/chat/completions",
            self.handle_chat_completions,
            methods=["POST"]
        )
        self.app.add_api_route(
            "/v1/models",
            self.handle_models,
            methods=["GET"]
        )
        if self.verbose:
            print("Listening on /v1/chat/completions")
            print("Listening on /v1/models")
        
        
    def count_active_chats(self) -> int:
        return len([v for k, v in self.active_chats.items() if v is not None and v.is_finished() is False])
        
        
    async def token_stream(self, chat: Chat, request: Request):
        handled = False
        try:
            first_sent = False
            while not chat.is_finished():
                if not first_sent:
                    first_sent = True
                    first_chunk = ChatCompletionChunk(chat)
                    first_json = first_chunk.json(first_chunk=True)
                    yield f"data: {json.dumps(first_json)}\n\n"
                else:
                    delta = await chat.get_delta()
                    if len(delta) > 0:
                        chunk = ChatCompletionChunk(chat)
                        json_data = chunk.json()
                        await chat.clear_delta()
                        yield f"data: {json.dumps(json_data)}\n\n"
                    elif chat.is_finished():
                        break
                await asyncio.sleep(0.05)
            if await chat.get_delta():
                last_chunk = ChatCompletionChunk(chat)
                last_json = last_chunk.json()
                await chat.clear_delta()
                yield f"data: {json.dumps(last_json)}\n\n"
            final_chunk = ChatCompletionChunk(chat)
            json_data = final_chunk.json(usage=True, final_chunk=True)
            yield f"data: {json.dumps(json_data)}\n\n"
            handled = True
            chat.print_stats()
        except asyncio.CancelledError:
            async with self.chats_lock:
                await chat.signal_stop(ChatStatus.COMPLETED, "stop")
            chat.print_stats()
            handled = True
            return
        finally:
            if not handled:
                async with self.chats_lock:
                    await chat.signal_stop(ChatStatus.STOPPED, "stop")
                chat.print_stats()
        
        
    async def handle_chat_completions(self, request: Request, chat_request: ChatCompletionsRequest):
        try:
            chat = Chat(chat_request)
            await self.chat_queue.put(chat)
            if chat.is_streaming():
                return StreamingResponse(self.token_stream(chat, request), media_type="text/event-stream")
            else:
                while not chat.is_finished():
                    await asyncio.sleep(0.5)
                chat.print_stats()
            return ChatCompletion(chat)
        except Exception as e:
            traceback.print_exc(limit=5)
        
        
    async def handle_models(self) -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model.config.model_dir,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "Meta"
                }
            ]
        }


    @lru_cache
    def get_chat_template(self) -> str:
        """
        Retrieves the chat template.

        Returns:
            str: The chat template.
        """
        if self.chat_template:
            return self.chat_template
        self.chat_template = self.tokenizer.tokenizer_config_dict.get('chat_template')
        if self.chat_template is None:
            self.chat_template = "{%- set ns = namespace(found=false) -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{%- set ns.found = true -%}{%- endif -%}{%- endfor -%}{%- if not ns.found -%}{{- 'You are a helpful assistant.' + '\n' -}}{%- endif %}{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '' + message['content'] + '\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'USER: ' + message['content'] + '\n'-}}{%- else -%}{{-'ASSISTANT: ' + message['content'] + '\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'ASSISTANT:'-}}{%- endif -%}"
        return self.chat_template


    @lru_cache
    def get_eos_token_ids(self) -> list[int]:
        eos_tokens = set()
        gen_eos_tids = self.tokenizer.config.generation_config.get('eos_token_id', None)
        
        if isinstance(gen_eos_tids, int):
            eos_tokens.add(gen_eos_tids)
            
        if isinstance(gen_eos_tids, list):
            for token_id in gen_eos_tids:
                eos_tokens.add(token_id)
        
        if self.tokenizer.eos_token_id:
            eos_tokens.add(self.tokenizer.eos_token_id)

        return list(eos_tokens)


    async def add_chat(self, chat: Chat):
        assert self.count_active_chats() < self.batch_size, "Batch size is at maximum"
        messages = chat.request.messages
        chat_template = self.get_chat_template()
        if not chat_template:
            raise ValueError("Chat template not found")
                        
        template = Template(chat_template)
        prompt = template.render({
            'messages': messages,
            'bos_token': self.tokenizer.bos_token,
            'eos_token': self.tokenizer.eos_token,
            'add_generation_prompt': True
        })
        if self.verbose:
            print(chat.id, prompt)
        
        sequence_tokens = self.tokenizer.encode(prompt).squeeze()
        chat.set_sequence_tokens(sequence_tokens)

        async with self.chats_lock:
            for i in range(self.batch_size):
                if not i in self.active_chats or self.active_chats[i] is None or self.active_chats[i].is_finished():
                    self.active_chats[i] = chat
                    break

    async def queue_chats(self):
        """
            Grab chats from queue and add to active batch when available.
        """
        with torch.inference_mode():
            while True:
                async with self.chats_lock:
                    num_active_chats = self.count_active_chats()
                
                if num_active_chats >= self.batch_size:
                    #  Too many active chats, sleep
                    await asyncio.sleep(0.5)
                    continue
                
                try:
                    new_chat = await self.chat_queue.get()
                    await self.add_chat(new_chat)
                    
                    gen_settings = ExLlamaV2Sampler.Settings()

                    if new_chat.request.top_p is not None:
                        gen_settings.top_p = new_chat.request.top_p

                    if new_chat.request.temperature is not None:
                        gen_settings.temperature = new_chat.request.temperature

                    if new_chat.request.top_k is not None:
                        gen_settings.top_k = new_chat.request.top_k

                    if new_chat.request.frequency_penalty is not None:
                        gen_settings.token_frequency_penalty = new_chat.request.frequency_penalty
                        
                    if new_chat.request.presence_penalty is not None:
                        gen_settings.token_presence_penalty = new_chat.request.presence_penalty
                    
                    job = ExLlamaV2DynamicJob(
                        input_ids = new_chat.sequence_tokens.unsqueeze(0),
                        max_new_tokens = 2048,
                        stop_conditions = self.get_eos_token_ids(),
                        gen_settings = gen_settings,
                    )
                    
                    new_chat.job = job
                    self.generator.enqueue(job)
                    
                except Exception:
                    traceback.print_exc(limit=5)


    async def process_chats(self):
        while self.abort_event is None or not self.abort_event.is_set():
            await asyncio.sleep(0.5)
            while self.generator.num_remaining_jobs():
                if self.abort_event is not None and self.abort_event.is_set():
                    break
                results = self.generator.iterate()
                for result in results:
                    job = result["job"]
                    async with self.chats_lock:
                        finished_chats = []
                        for i, chat in self.active_chats.items():
                            if chat is None or chat.is_finished(): continue
                            if chat.job == job:
                                if 'eos' in result and result['eos'] is True:
                                    eos_reason = result['eos_reason']
                                    finish_reason = "stop"
                                    chat_status = ChatStatus.COMPLETED
                                    if eos_reason == "max_new_tokens":
                                        finish_reason = "length"
                                        chat_status = ChatStatus.STOPPED
                                    await chat.signal_stop(chat_status, finish_reason)
                                    finished_chats.append(i)
                                if 'text' in result:
                                    await chat.add_delta(result['text'])
                                    await asyncio.sleep(0.01)
                        if len(finished_chats) > 0:
                            for i in finished_chats:
                                self.active_chats[i] = None


    def on_startup(self):
        asyncio.get_event_loop().create_task(self.queue_chats())
        asyncio.get_event_loop().create_task(self.process_chats())


    def load(self):
        
        max_chunk_size = 512
        total_context = max_chunk_size * 16 + (self.batch_size * max_chunk_size * 2)
        
        self.config = ExLlamaV2Config(model_dir=self.model_path)
        self.config.max_input_len = max_chunk_size
        self.config.max_attention_size = max_chunk_size ** 2
        self.config.prepare()
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=total_context, lazy=True)
        self.model.load_autosplit(cache=self.cache, progress=True)
        
        self.generator = ExLlamaV2DynamicGenerator(
            model = self.model,
            cache = self.cache,
            tokenizer = self.tokenizer,
            max_batch_size = self.batch_size,
            max_chunk_size = max_chunk_size,
            paged = True
        )
        
        self.generator.warmup()

        
    def handle_interrupt(self, signum, frame):
        self.abort_event.set()


    def serve(self):
        print(f" -- Starting OpenAI-compatible server on {self.ip} port {self.port}")
        signal.signal(signal.SIGINT, self.handle_interrupt)
        self.load()
        self.app = FastAPI(on_startup=[self.on_startup])
        self.add_routes()
        uvicorn.run(self.app, host=self.ip, port=self.port)
