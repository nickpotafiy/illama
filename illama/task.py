import asyncio
import time
import uuid

from enum import Enum

from exllamav2.generator.dynamic import ExLlamaV2DynamicJob


class TaskStatus(Enum):
    QUEUED = ("QUEUED",)
    PROCESSING = ("PROCESSING",)
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


class Task:

    def __init__(self):
        self.created_at = time.time()
        self.id: uuid.UUID = uuid.uuid4()
        self.status: TaskStatus = TaskStatus.QUEUED
        self._lock = asyncio.Lock()
        self._abort = asyncio.Event()
        self.job: ExLlamaV2DynamicJob = None

    def set_status(self, status: TaskStatus):
        self.status = status

    def is_finished(self) -> bool:
        if self._abort.is_set():
            self.set_status(TaskStatus.STOPPED)
            return True
        return self.status in {TaskStatus.COMPLETED, TaskStatus.STOPPED}
