import asyncio
from enum import Enum
import time
import uuid

from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F

from exllamav2.model import ExLlamaV2
from exllamav2.tokenizer.tokenizer import ExLlamaV2Tokenizer

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
    max_tokens: Optional[int] = 4096
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str], List[int]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Union[str, bool]]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
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


    def get_sequence_length(self) -> int:
        """Get the number of tokens currently in the sequence."""
        if self.sequence_tokens is None:
            return 0
        seq_len = self.sequence_tokens.shape[0]
        return seq_len


    async def get_delta(self) -> str:
        async with self._lock:
            return self.delta
        
        
    async def clear_delta(self):
        async with self._lock:
            self.delta = ""
            
            
    def padded_sequence(self, max_length: int, tokenizer: ExLlamaV2Tokenizer) -> torch.LongTensor:
        return F.pad(self.sequence_tokens, (0, max_length - len(self.sequence_tokens)),
                     value=tokenizer.pad_token_id)
            
            
    async def sample_next_token(self, logits, tokenizer: ExLlamaV2Tokenizer, model: ExLlamaV2) -> bool:
        next_token_id = process_logits_and_sample(logits, len(self.sequence_tokens))
        eos_tokens = set()
        generation_tokens = tokenizer.config.generation_config.get('eos_token_id', None)
        
        if isinstance(generation_tokens, int):
            eos_tokens.add(generation_tokens)
            
        if isinstance(generation_tokens, list):
            for generation_token in generation_tokens:
                eos_tokens.add(generation_token)
        
        if tokenizer.eos_token_id:
            eos_tokens.add(tokenizer.eos_token_id)
        
        if next_token_id.item() in eos_tokens:
            self.finish_reason = "stop"
            self.status = ChatStatus.COMPLETED
            return True
        
        next_token_str = tokenizer.decode(next_token_id, decode_special_tokens=True)
        piece = tokenizer.tokenizer_model.id_to_piece(next_token_id.item())
        if piece and piece[0] == "▁":
            next_token_str = " " + next_token_str
        stop_tokens = getattr(self.request, "stop", [])
        
        if stop_tokens is not None and next_token_str in stop_tokens:
            self.finish_reason = "stop"
            self.status = ChatStatus.COMPLETED
            return True

        async with self._lock:
            self.sequence_tokens = torch.cat([self.sequence_tokens, next_token_id])
            self.delta += next_token_str
            self.completion_tokens += 1

        if self.completion_tokens + self.prompt_tokens >= model.config.max_seq_len - 1:
            self.finish_reason = "length"
            self.status = ChatStatus.COMPLETED
            return True

        max_tokens = getattr(self.request, 'max_tokens', None)
        if self.completion_tokens >= max_tokens:
            self.finish_reason = "length"
            self.status = ChatStatus.COMPLETED
            return True

        return False


    async def signal_stop(self, status: ChatStatus, finish_reason: str):
        self.status = status
        self.finish_reason = finish_reason

class ChatObject:


    def __init__(self, chat: Chat, object_name: str):
        self.chat = chat
        self.object_name = object_name
        self.created = int(time.time())


    def json(self, usage: bool = False) -> dict:
        dict = {
            'id': str(self.chat.id),
            'object': self.object_name,
            'created': self.created,
            'model': getattr(self.chat.request, 'model', None),
            'choices': [{
                'index': 0,
                'finish_reason': self.chat.finish_reason if self.chat.finish_reason else ""
            }]
        }
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


    def json(self, usage: bool = False) -> dict:
        json = super().json(usage)
        if self.chat.is_finished():
            # Last chunk
            json['choices'][0]['delta'] = {}
        elif self.chat.completion_tokens == 0:
            # First chunk
            json['choices'][0]['delta'] = {
                'role': 'assistant',
                'content': ''
            }
        else:
            # Every other chunk
            json['choices'][0]['delta'] = {
                'content': self.chat.delta
            }
        return json


def apply_temp(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    if temperature != 0:
        logits = logits / temperature
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply top-p (nucleus) sampling to logits."""
    # Sort the logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Compute stable softmax probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Calculate the cumulative probabilities of the sorted probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask to identify indices where cumulative probability exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p

    # Create a new mask tensor and shift the values
    shifted_indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    shifted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    shifted_indices_to_remove[..., 0] = 0

    # Set logits corresponding to indices to be removed to -inf
    sorted_logits[shifted_indices_to_remove] = float('-inf')

    # Create a tensor full of -inf of the same shape as logits
    updated_logits = torch.full_like(logits, float('-inf'))

    # Scatter the sorted logits back to the updated_logits tensor
    updated_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    
    return updated_logits


def apply_top_k(logits: torch.Tensor, top_k: int = 40):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        
    return logits


def apply_repeat_penalty(logits, generated_tokens, penalty):
    if penalty != 1.0:
        for token in set(generated_tokens):
            logits[token] /= penalty
            
    return logits


def process_logits_and_sample(
        last_token_logits: torch.Tensor,
        sequence_length: int,
        generated_tokens: torch.Tensor = None,
        temp: float = None,
        top_p: float = None,
        top_k: int = None,
        repeat_penalty: float = None) -> torch.Tensor:
    # Apply temperature
    if temp:
        last_token_logits = apply_temp(last_token_logits, temp)

    # Apply top_p
    if top_p:
        last_token_logits = apply_top_p(last_token_logits, top_p)

    # Apply top_k
    if top_k:
        last_token_logits = apply_top_k(last_token_logits, top_k)

    # Apply repeat penalty
    if generated_tokens and repeat_penalty:
        last_token_logits = apply_repeat_penalty(
            last_token_logits, generated_tokens, repeat_penalty)

    # Apply softmax to get probabilities
    last_token_probs = F.softmax(last_token_logits, dim=-1)

    # Sample a token from the probability distribution
    token = torch.multinomial(last_token_probs, 1)

    return token