import signal
import threading
import time
import traceback
import asyncio
import json
import torch
import uvicorn

from functools import lru_cache
from typing import Dict
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from jinja2 import Template

from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer, ExLlamaV2Config

from exllamav2.generator.dynamic import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob
from exllamav2.generator.sampler import ExLlamaV2Sampler

from illama.oai import (
    ChatCompletionsTask,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatCompletionsRequest,
    Task,
    TaskStatus,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsTask,
)


class IllamaServer:
    """
    Initialize Illama server

    Args:
        ip (str): The IP address of the server.
        port (int): The port number of the server.
        model_path (str): The path to the model to serve.
        batch_size (int, optional): The batch size for processing tasks. Defaults to 5.
        max_chunk_size (int, optional): Maximum number of tokens to process in parallel during prefill (prompt ingestion). Should not
            exceed the model's max_input_len but can be lowered to trade off prompt speed for a shorter
            interruption to ongoing jobs when a new job is started. Defaults to 256.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    def __init__(
        self,
        ip: str,
        port: int,
        model_path: str,
        batch_size: int = 5,
        max_chunk_size: int = 256,
        verbose: bool = False,
    ):
        self.ip: str = ip
        self.port: int = port
        self.model_path: str = model_path
        self.batch_size: int = batch_size
        self.max_chunk_size = max_chunk_size
        self.verbose: bool = verbose

        self.model: ExLlamaV2 = None
        self.tokenizer: ExLlamaV2Tokenizer = None
        self.cache: ExLlamaV2Cache = None

        self.tasks_lock = asyncio.Lock()
        self.task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self.active_tasks: Dict[int, Task] = {}

        self.chat_template: str = None
        self.abort_event = threading.Event()
        signal.signal(signal.SIGINT, self.handle_interrupt)

        self.running: bool = False

    def add_routes(self):
        self.app.add_api_route(
            "/v1/chat/completions", self.handle_chat_completions, methods=["POST"]
        )
        self.app.add_api_route(
            "/v1/embeddings", self.handle_embeddings, methods=["POST"]
        )
        self.app.add_api_route("/v1/models", self.handle_models, methods=["GET"])
        if self.verbose:
            print("Listening on /v1/chat/completions")
            print("Listening on /v1/models")
            print("Listening on /v1/embeddings")

    def count_active_tasks(self) -> int:
        return len(
            [
                v
                for k, v in self.active_tasks.items()
                if v is not None and v.is_finished() is False
            ]
        )

    async def chat_completion_token_stream(self, task: ChatCompletionsTask):
        handled = False
        try:
            first_sent = False
            while not task.is_finished() and (
                self.abort_event is None or not self.abort_event.is_set()
            ):
                if not first_sent:
                    first_sent = True
                    first_chunk = ChatCompletionChunk(task)
                    first_json = first_chunk.json(first_chunk=True)
                    yield f"data: {json.dumps(first_json)}\n\n"
                else:
                    delta = await task.get_delta()
                    if len(delta) > 0:
                        chunk = ChatCompletionChunk(task)
                        json_data = chunk.json()
                        await task.clear_delta()
                        yield f"data: {json.dumps(json_data)}\n\n"
                    elif task.is_finished():
                        break
                await asyncio.sleep(0.05)
            if await task.get_delta():
                last_chunk = ChatCompletionChunk(task)
                last_json = last_chunk.json()
                await task.clear_delta()
                yield f"data: {json.dumps(last_json)}\n\n"
            final_chunk = ChatCompletionChunk(task)
            json_data = final_chunk.json(usage=True, final_chunk=True)
            yield f"data: {json.dumps(json_data)}\n\n"
            handled = True
            task.print_stats()
        except asyncio.CancelledError:
            async with self.tasks_lock:
                await task.signal_stop(TaskStatus.COMPLETED, "stop")
            task.print_stats()
            handled = True
            return
        finally:
            if not handled:
                async with self.tasks_lock:
                    await task.signal_stop(TaskStatus.STOPPED, "stop")
                task.print_stats()

    async def add_task(self, task: Task):
        async with self.tasks_lock:
            await self.task_queue.put(task)

    async def wait_for(self, task: Task):
        while self.abort_event is None or not self.abort_event.is_set():
            await asyncio.sleep(0.1)
            async with task._lock:
                if task.is_finished():
                    break

    async def add_and_wait(self, task: Task):
        await self.add_task(task)
        await self.wait_for(task)

    async def handle_embeddings(self, request: EmbeddingsRequest):
        task = EmbeddingsTask(request)
        await self.add_and_wait(task)
        response = EmbeddingsResponse(task, request)
        json = response.json()
        return json

    async def handle_chat_completions(self, request: ChatCompletionsRequest):
        task = ChatCompletionsTask(request)
        await self.add_task(task)
        if task.is_streaming():
            return StreamingResponse(
                self.chat_completion_token_stream(task), media_type="text/event-stream"
            )
        else:
            await self.wait_for(task)
            task.print_stats()
        response = ChatCompletionResponse(task)
        return response.json(usage=True)

    async def handle_models(self) -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model.config.model_dir,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "Meta",
                }
            ],
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
        self.chat_template = self.tokenizer.tokenizer_config_dict.get("chat_template")
        if self.chat_template is None:
            self.chat_template = "{%- set ns = namespace(found=false) -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{%- set ns.found = true -%}{%- endif -%}{%- endfor -%}{%- if not ns.found -%}{{- 'You are a helpful assistant.' + '\n' -}}{%- endif %}{%- for message in messages %}{%- if message['role'] == 'system' -%}{{- '' + message['content'] + '\n' -}}{%- else -%}{%- if message['role'] == 'user' -%}{{-'USER: ' + message['content'] + '\n'-}}{%- else -%}{{-'ASSISTANT: ' + message['content'] + '\n' -}}{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{-'ASSISTANT:'-}}{%- endif -%}"
        return self.chat_template

    @lru_cache
    def get_eos_token_ids(self) -> list[int]:
        eos_tokens = set()
        gen_eos_tids = self.tokenizer.config.generation_config.get("eos_token_id", None)

        if isinstance(gen_eos_tids, int):
            eos_tokens.add(gen_eos_tids)

        if isinstance(gen_eos_tids, list):
            for token_id in gen_eos_tids:
                eos_tokens.add(token_id)

        if self.tokenizer.eos_token_id:
            eos_tokens.add(self.tokenizer.eos_token_id)

        return list(eos_tokens)

    def stop(self):
        self.abort_event.set()

    async def queue_chats(self):
        """
        Grab chats from queue and add to active batch when available.
        """
        with torch.inference_mode():
            while self.abort_event is None or not self.abort_event.is_set():
                async with self.tasks_lock:
                    num_active_tasks = self.count_active_tasks()

                if num_active_tasks >= self.batch_size:
                    #  Too many active tasks, sleep
                    await asyncio.sleep(0.25)
                    continue

                try:
                    task = await self.task_queue.get()

                    if isinstance(task, ChatCompletionsTask):

                        messages = task.request.messages
                        chat_template = self.get_chat_template()

                        assert chat_template is not None, "Chat template not found"
                        assert (
                            self.count_active_tasks() < self.batch_size
                        ), "Too many active tasks, likely active_tasks modified without lock"

                        template = Template(chat_template)

                        prompt = template.render(
                            {
                                "messages": messages,
                                "bos_token": self.tokenizer.bos_token,
                                "eos_token": self.tokenizer.eos_token,
                                "add_generation_prompt": True,
                            }
                        )

                        if self.verbose:
                            print(task.id, prompt)

                        sequence_tokens = self.tokenizer.encode(prompt).squeeze()
                        task.set_sequence_tokens(sequence_tokens)

                        gen_settings = ExLlamaV2Sampler.Settings()

                        if task.request.top_p is not None:
                            gen_settings.top_p = task.request.top_p
                        if task.request.temperature is not None:
                            gen_settings.temperature = task.request.temperature
                        if task.request.top_k is not None:
                            gen_settings.top_k = task.request.top_k
                        if task.request.frequency_penalty is not None:
                            gen_settings.token_frequency_penalty = (
                                task.request.frequency_penalty
                            )
                        if task.request.presence_penalty is not None:
                            gen_settings.token_presence_penalty = (
                                task.request.presence_penalty
                            )

                        job = ExLlamaV2DynamicJob(
                            input_ids=task.sequence_tokens.unsqueeze(0),
                            max_new_tokens=task.request.max_tokens,
                            stop_conditions=self.get_eos_token_ids(),
                            gen_settings=gen_settings,
                        )
                        task.job = job
                        self.generator.enqueue(job)

                    elif isinstance(task, EmbeddingsTask):
                        gen_settings = ExLlamaV2Sampler.Settings()
                        input_ids = self.tokenizer.encode(task.request.input)
                        job = ExLlamaV2DynamicJob(
                            input_ids=input_ids,
                            max_new_tokens=0,
                            gen_settings=gen_settings,
                            return_hidden_state=True,
                        )
                        task.job = job
                        self.generator.enqueue(job)
                    else:
                        print("Unhandled task:", type(task))
                        continue

                    # add task to active tasks

                    for i in range(self.batch_size):
                        async with self.tasks_lock:
                            if (
                                not i in self.active_tasks
                                or self.active_tasks[i] is None
                                or self.active_tasks[i].is_finished()
                            ):
                                self.active_tasks[i] = task
                                break

                except Exception:
                    traceback.print_exc(limit=5)

    async def process_chats(self):
        while self.abort_event is None or not self.abort_event.is_set():
            await asyncio.sleep(0.5)
            while self.generator.num_remaining_jobs() and (
                self.abort_event is None or not self.abort_event.is_set()
            ):
                await asyncio.sleep(0.005)
                if self.abort_event is not None and self.abort_event.is_set():
                    break
                results = self.generator.iterate()
                for result in results:
                    if self.abort_event is not None and self.abort_event.is_set():
                        return
                    job = result["job"]
                    async with self.tasks_lock:
                        finished_chats = []
                        for i, task in self.active_tasks.items():
                            if task is None or task.is_finished():
                                continue
                            if task.job == job:
                                if isinstance(task, ChatCompletionsTask):
                                    if "eos" in result and result["eos"] is True:
                                        eos_reason = result["eos_reason"]
                                        finish_reason = "stop"
                                        chat_status = TaskStatus.COMPLETED
                                        if eos_reason == "max_new_tokens":
                                            finish_reason = "length"
                                            chat_status = TaskStatus.STOPPED
                                        await task.signal_stop(
                                            chat_status, finish_reason
                                        )
                                        finished_chats.append(i)
                                    if "text" in result:
                                        await task.add_delta(result["text"])
                                        await asyncio.sleep(0.01)
                                elif isinstance(task, EmbeddingsTask):
                                    if "eos" in result and result["eos"] is True:
                                        if "hidden_state" in result:
                                            task.hidden_state = result[
                                                "hidden_state"
                                            ].squeeze()
                                        finish_reason = "stop"
                                        chat_status = TaskStatus.COMPLETED
                                        task.set_status(chat_status)
                                        finished_chats.append(i)
                                        await asyncio.sleep(0.01)
                                else:
                                    print("Unhandled task type:", type(task))
                                    continue
                        if len(finished_chats) > 0:
                            for i in finished_chats:
                                self.active_tasks[i] = None

    def on_startup(self):
        asyncio.get_event_loop().create_task(self.queue_chats())
        asyncio.get_event_loop().create_task(self.process_chats())

    def load(self):

        max_chunk_size = self.max_chunk_size
        total_context = max_chunk_size * 16 + (self.batch_size * max_chunk_size * 8)

        self.config = ExLlamaV2Config(model_dir=self.model_path)
        self.config.max_input_len = max_chunk_size
        self.config.max_attention_size = max_chunk_size**2
        self.config.prepare()
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=total_context, lazy=True)
        self.model.load_autosplit(cache=self.cache, progress=True)

        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
            max_batch_size=self.batch_size,
            max_chunk_size=max_chunk_size,
            paged=True,
        )

        self.generator.warmup()
        self.running = True

    def handle_interrupt(self, signum, frame):
        self.abort_event.set()

    def serve(self):
        print(f":: Starting OpenAI-compatible server on {self.ip} port {self.port}")
        self.load()
        self.app = FastAPI(on_startup=[self.on_startup])
        self.add_routes()
        uvicorn.run(self.app, host=self.ip, port=self.port)
