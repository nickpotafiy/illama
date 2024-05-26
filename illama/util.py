import asyncio
import time
import uuid
import torch
import torch.nn.functional as F

from enum import Enum
from pydantic import BaseModel
from typing import Dict, List, Optional, Union

from exllamav2.generator.dynamic import ExLlamaV2DynamicJob


class ChatStatus(Enum):
    QUEUED = "QUEUED",
    PROCESSING = "PROCESSING",
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


class ChatFinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"


class Message(BaseModel):
    role: str = None
    content: str = None


class ChatCompletionsRequest(BaseModel):
    messages: List[Message]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = 2048
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str], List[int]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Union[str, bool]]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    tools: Optional[List[Dict[str, Union[str, Dict]]]] = None
    tool_choice: Optional[Union[str, Dict[str, Union[str, Dict]]]] = None
    user: Optional[str] = None


class Chat:


    def __init__(self, request: ChatCompletionsRequest):
        self.id: uuid.UUID = uuid.uuid4()
        self.start_time = time.time()
        self.finish_reason = None
        self.status: ChatStatus = ChatStatus.QUEUED
        self.output: str = ""
        self.delta: str = ""
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.request: ChatCompletionsRequest = request
        self.sequence_tokens: torch.LongTensor = None
        self._lock = asyncio.Lock()
        self._abort = asyncio.Event()
        self.preprocessed = False
        self.job: ExLlamaV2DynamicJob = None


    def is_streaming(self) -> bool:
        """Checks if the current chat is in streaming mode."""
        return getattr(self.request, 'stream', False)


    def is_finished(self) -> bool:
        if self._abort.is_set():
            self.status = ChatStatus.STOPPED
            self.finish_reason = "stop"
            return True
        
        return self.status in {ChatStatus.COMPLETED, ChatStatus.STOPPED}


    def tokens_per_second(self) -> float:
        return self.completion_tokens / (time.time() - self.start_time)


    def set_sequence_tokens(self, sequence_tokens: torch.LongTensor = None):
        self.sequence_tokens = sequence_tokens
        self.prompt_tokens = self.sequence_tokens.shape[0]


    async def add_delta(self, delta: str):
        async with self._lock:
            self.delta += delta
            self.completion_tokens += 1
            

    async def get_delta(self) -> str:
        async with self._lock:
            return self.delta
        
        
    async def clear_delta(self):
        async with self._lock:
            self.delta = ""
            

    async def signal_stop(self, status: ChatStatus, finish_reason: str):
        async with self._lock:
            self.status = status
            self.finish_reason = finish_reason


    def print_stats(self):
        print(self.id, self.prompt_tokens, "prompt,", self.completion_tokens, 
              "completion,", round(self.tokens_per_second(), 2), "tok/s,", "finished", self.finish_reason)


class ChatObject:


    def __init__(self, chat: Chat, object_name: str):
        self.chat = chat
        self.object_name = object_name
        self.created = int(time.time())


    def json(self, usage: bool = False, finish_reason: bool = False) -> dict:
        dict = {
            'id': str(self.chat.id),
            'object': self.object_name,
            'created': self.created,
            'model': getattr(self.chat.request, 'model', None),
            'choices': [{
                'index': 0,
            }]
        }
        
        if finish_reason:
            dict['choices'][0]['finish_reason'] = getattr(self.chat, 'finish_reason', "")
            
        if usage:
            dict["usage"] = {
                "completion_tokens": self.chat.completion_tokens,
                "prompt_tokens": self.chat.prompt_tokens,
                "total_tokens": self.chat.prompt_tokens + self.chat.completion_tokens
            }
        return dict


class ChatCompletion(ChatObject):


    def __init__(self, chat: Chat):
        super().__init__(chat, "chat.completion")


    def json(self, usage: bool = False) -> dict:
        json = super().json(usage)
        json['choices'][0]['message'] = {
            'role': 'assistant',
            'content': getattr(self.chat, 'output', None)
        }
        return json


class ChatCompletionChunk(ChatObject):


    def __init__(self, chat: Chat):
        super().__init__(chat, "chat.completion.chunk")


    def json(self, usage: bool = False, first_chunk: bool = False, final_chunk: bool = False) -> dict:
        json = super().json(usage=usage, finish_reason=final_chunk)        
        if first_chunk:
            json['choices'][0]['delta'] = {
                'role': 'assistant',
                'content': ''
            }
        elif final_chunk:
            json['choices'][0]['delta'] = {}
        else:
            json['choices'][0]['delta'] = {
                'content': self.chat.delta
            }
        return json
