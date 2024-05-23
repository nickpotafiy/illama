from functools import lru_cache
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from jinja2 import Template

import torch
import uvicorn
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)

import asyncio
import json

from illama.util import (
    Chat,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionsRequest,
    ChatStatus
)


class ILlamaServer:


    def __init__(self, ip: str, port: int, model: ExLlamaV2, tokenizer: ExLlamaV2Tokenizer, cache: ExLlamaV2Cache, verbose: bool = False):
        self.ip: str = ip
        self.port: int = port
        self.model: ExLlamaV2 = model
        self.tokenizer: ExLlamaV2Tokenizer = tokenizer
        self.cache: ExLlamaV2Cache = cache
        self.verbose = verbose

        self.chat_template: str = None
        self.chats_lock = asyncio.Lock()
        self.chat_queue: asyncio.Queue[Chat] = asyncio.Queue()
        self.active_chats: list[Chat] = []
        self.chat_template = None
        self.zero_out_cache()
        
        
    def zero_out_cache(self):
        if self.cache:
            with torch.inference_mode():
                for i in range(self.model.config.num_hidden_layers):
                    k, v = self.cache.get_kv_state(i, 0, 0, 0)
                    kv = k.narrow(1, 0, self.cache.max_seq_len).narrow(0, 0, self.cache.batch_size)
                    vv = v.narrow(1, 0, self.cache.max_seq_len).narrow(0, 0, self.cache.batch_size)
                    kv.copy_(torch.zeros_like(kv))
                    vv.copy_(torch.zeros_like(vv))
        
        
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
        return len(self.active_chats)
        
        
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


    async def add_chat(self, chat: Chat):
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
        
        sequence_tokens = self.tokenizer.encode(prompt).to('cuda').squeeze()
        chat.set_sequence_tokens(sequence_tokens)

        async with self.chats_lock:
            index = len(self.active_chats)
            self.active_chats.append(chat)
            # zero out cache row, might not be necessary
            for layer in range(self.model.config.num_hidden_layers):
                key_states, value_states = self.cache.get_kv_state(layer, self.cache.batch_size, 0, 0)
                key_states.narrow(1, 0, self.cache.max_seq_len).narrow(0, index, 1).fill_(0.0)
                value_states.narrow(1, 0, self.cache.max_seq_len).narrow(0, index, 1).fill_(0.0)

    async def queue_chats(self):
        """
            Grab chats from queue and add them to active chats for processing if conditions are met.
        """
        with torch.inference_mode():
            while True:
                async with self.chats_lock:
                    num_active_chats = self.count_active_chats()
                
                if num_active_chats >= self.cache.batch_size:
                    #  Too many active chats, sleep
                    await asyncio.sleep(0.5)
                    continue
                
                try:
                    new_chat = await self.chat_queue.get()
                    await self.add_chat(new_chat)

                except Exception:
                    traceback.print_exc(limit=5)
    
    
    def get_max_seq_length(self) -> int:
        return max([len(x.sequence_tokens) for x in self.active_chats] + [0])
    
    def get_max_seq_length_except(self, exclude_chat: Chat) -> int:
        return max([len(x.sequence_tokens) for x in self.active_chats if x is not exclude_chat] + [0])
        

    async def preprocess(self, chat: Chat):
        chat_idx = self.active_chats.index(chat)
        # use temp cache to pre-process the initial sequence and obtain its kv outputs
        temp_cache = ExLlamaV2Cache(self.model, batch_size=1, max_seq_len=chat.get_sequence_length())
        
        input_ids = chat.sequence_tokens.unsqueeze(0)

        self.model.forward(
            input_ids=input_ids,
            cache=temp_cache,
            last_id_only=False,
            preprocess_only=True,
            abort_event=chat._abort
        )
        
        chat.prompt_tokens = chat.get_sequence_length()
        max_length = self.get_max_seq_length()
        cache_seq_len = self.cache.current_seq_len
        
        if cache_seq_len > 0 and cache_seq_len < max_length:
            cache_offset = max_length - cache_seq_len
            # shift active chats cache right
            self.copy_cache(self.cache, self.cache, 0, self.cache.current_seq_len, 0, len(self.active_chats), cache_offset, self.cache.current_seq_len, 0, len(self.active_chats))

        #  copy temp cache to perm cache ensuring end-alignment of last token with rest of batch
        kv_chat_offset = max_length - chat.get_sequence_length()
        
        self.copy_cache(temp_cache, self.cache, 0, temp_cache.current_seq_len, 0, 1, kv_chat_offset, temp_cache.current_seq_len, chat_idx, 1)
        
        if temp_cache.current_seq_len > self.cache.current_seq_len:
            self.cache.current_seq_len = temp_cache.current_seq_len
        chat.preprocessed = True
        return temp_cache

    def copy_cache(self, cache_from: ExLlamaV2Cache, cache_to: ExLlamaV2Cache, from_token, from_tokens, from_batch, from_batches, to_token, to_tokens, to_batch, to_batches):
        for layer in range(self.model.config.num_hidden_layers):
            from_key_states, from_value_states = cache_from.get_kv_state(layer, from_batches, from_token, from_token + from_tokens)
            to_key_states, to_value_states = cache_to.get_kv_state(layer, to_batches, to_token, to_token + to_tokens)
            from_key_view = from_key_states.narrow(1, from_token, from_tokens).narrow(0, from_batch, from_batches)
            from_value_view = from_value_states.narrow(1, from_token, from_tokens).narrow(0, from_batch, from_batches)
            to_key_view = to_key_states.narrow(1, to_token, to_tokens).narrow(0, to_batch, to_batches)
            to_value_view = to_value_states.narrow(1, to_token, to_tokens).narrow(0, to_batch, to_batches)
            to_key_view.copy_(from_key_view)
            to_value_view.copy_(from_value_view)
            

    async def batch_forward(self):
        max_length = self.get_max_seq_length()    
                        
        last_tokens = []
        attn_masks = []
        pos_offsets = []
        
        for i, chat in enumerate(self.active_chats):
            last_tokens.append(chat.sequence_tokens[-1].unsqueeze(0))
            mask = torch.full((max_length,), float('0.0'), dtype=torch.float16)
            mask[:max_length - chat.get_sequence_length()] = float('-inf')
            attn_masks.append(mask)
            offset = - 0 - (max_length - chat.get_sequence_length())
            pos_offsets.append(torch.IntTensor([offset]))
        
        input_ids = torch.stack(last_tokens)
        attention_mask = torch.stack(attn_masks)        
        position_offsets = torch.stack(pos_offsets)

        outputs = self.model.forward(
            input_ids=input_ids,
            input_mask=attention_mask,
            cache=self.cache,
            last_id_only=True,
            position_offsets=position_offsets
        )

        remove_chats = []
        for batch, output in enumerate(outputs):
            chat = self.active_chats[batch]
            last_token_logits = output[0]
            finished = await chat.sample_next_token(last_token_logits, self.tokenizer, self.model)
            if finished:
                remove_chats.append(chat)
        return remove_chats


    async def process_chats(self):
        with torch.inference_mode():
            remove_chats = []
            while True:
                await asyncio.sleep(0.01)
                no_chats = False                
                async with self.chats_lock:
                    if self.count_active_chats() == 0:
                        no_chats = True
                if no_chats:
                    await asyncio.sleep(0.5)
                    continue
                
                async with self.chats_lock:  # chats need to be locked for this operation
                    go_to_start = False
                    for chat in self.active_chats:
                        if chat.is_finished() and chat in self.active_chats and not chat in remove_chats:
                            remove_chats.append(chat)
                            break
                        if chat and not chat.preprocessed:
                            await self.preprocess(chat)                            
                            go_to_start = True
                        continue
                    if len(remove_chats) > 0:
                        active_chats = self.count_active_chats()
                        old_seq_len = self.get_max_seq_length()
                        for remove_chat in remove_chats:
                            active_chats = self.count_active_chats()
                            chat_idx = self.active_chats.index(remove_chat)
                            # if there's multiple active chats, we need to possibly shift cache
                            if active_chats > 1:
                                # check if the current chat is the last active one
                                last_chat_index = len(self.active_chats) - 1
                                last_active_chat = self.active_chats[last_chat_index]
                                if not last_active_chat == remove_chat:
                                    # swap last chat with current one
                                    self.active_chats[chat_idx] = last_active_chat
                                    del self.active_chats[last_chat_index] # remove the last chat
                                    # shift the last chat's cache to the current one
                                    self.copy_cache(self.cache, self.cache, 0, old_seq_len, last_chat_index, 1, 0, old_seq_len, chat_idx, 1)
                                else:
                                    del self.active_chats[chat_idx]
                            else:
                                del self.active_chats[chat_idx]
                        new_seq_len = self.get_max_seq_length()
                        if old_seq_len > new_seq_len:
                            # shift active cache batches left
                            offset = old_seq_len - new_seq_len
                            for i, chat in enumerate(self.active_chats):
                                # create temp cache to store current chat sequence cache
                                temp_cache = ExLlamaV2Cache(self.model, batch_size=1, max_seq_len=chat.get_sequence_length())
                                # move chat cache from perm to temp
                                chat_seq = chat.get_sequence_length()
                                self.copy_cache(self.cache, temp_cache, offset, chat_seq, i, 1, 0, chat_seq, 0, 1)
                                # move from temp back to perm, remapping to beginning
                                self.copy_cache(temp_cache, self.cache, 0, chat_seq, 0, 1, 0, chat_seq, i, 1)
                                
                        self.cache.current_seq_len = new_seq_len
                        remove_chats.clear()
                        go_to_start = True
                        
                    if go_to_start:
                        continue
                    
                    chats_to_remove = await self.batch_forward()
                    if len(chats_to_remove) > 0:
                        for remove in chats_to_remove:
                            remove_chats.append(remove)
                    
    def on_startup(self):
        asyncio.get_event_loop().create_task(self.queue_chats())
        asyncio.get_event_loop().create_task(self.process_chats())

    def serve(self):
        print(f" -- Starting OpenAI-compatible server on {self.ip} port {self.port}")
        self.app = FastAPI(on_startup=[self.on_startup])
        self.add_routes()
        uvicorn.run(self.app, host=self.ip, port=self.port)
