"""Task-scoped Core source-layer IDs; no attention, mask, or velocity tensors are cached."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskCoreLayerCache:
    count: int = 7
    task: str | None = None
    blocks: tuple[int, ...] | None = None
    completed_chunks: int = 0

    def begin_task(self, task: str, *, reset: bool = False):
        if not isinstance(task, str) or not task:
            raise ValueError("Task identity must be a nonempty string")
        if reset or task != self.task:
            self.task = task
            self.blocks = None
            self.completed_chunks = 0

    def commit(self, blocks):
        """Commit only after successful generation; a failed first chunk cannot publish a stale plan."""
        if self.task is None:
            raise RuntimeError("begin_task is required before committing a chunk")
        value = tuple(blocks)
        if (
            len(value) != self.count
            or len(set(value)) != self.count
            or any(type(b) is not int or not 0 <= b < 28 for b in value)
        ):
            raise ValueError("Expected unique valid Core layer IDs")
        if self.blocks is not None and value != self.blocks:
            raise RuntimeError("Core layer IDs/order changed inside the same task")
        self.blocks = value
        self.completed_chunks += 1
