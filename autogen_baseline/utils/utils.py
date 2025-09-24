import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

from autogen_agentchat.agents import MessageFilterAgent
from autogen_agentchat.messages import BaseChatMessage
from autogen_core.logging import LLMCallEvent


def my_parse_r1_content(content: str) -> tuple[str | None, str]:
    """Parse the content of an R1-style message that contains a `<think>...</think>` field.
    Qwen/Qwen3-235B-A22B-Thinking-2507 output doesn't include the opening <think> tag, so we
    handle that case by inserting a <think> tag at the start if we see a </think> without a <think>."""

    # Find the start and end of the think field
    think_start = content.find("<think>")
    think_end = content.find("</think>")

    if think_start == -1 and think_end > -1:
        warnings.warn(
            "Model response content contains a </think> but no <think>. Inserting a <think> at the beginning...",
            UserWarning,
            stacklevel=2,
        )

    if think_end == -1:
        warnings.warn(
            "Could not find </think> in model response content. " "No thought was extracted.",
            UserWarning,
            stacklevel=2,
        )
        return None, content

    if think_end < think_start:
        warnings.warn(
            "Found </think> before <think> in model response content. " "No thought was extracted.",
            UserWarning,
            stacklevel=2,
        )
        return None, content

    # Extract the think field
    thought = content[think_start + len("<think>") : think_end].strip()

    # Extract the rest of the content, skipping the think field.
    content = content[think_end + len("</think>") :].strip()

    return thought, content


class LLMUsageTracker(logging.Handler):
    def __init__(self, log_file_path: str | Path | None = None) -> None:
        """Logging handler that tracks the number of tokens used in the prompt and completion."""
        super().__init__()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._log_file_path = Path(log_file_path) if log_file_path else None
        self._log_file = None

        # Open the log file if path is provided
        if self._log_file_path:
            self._log_file = open(self._log_file_path, "a", encoding="utf-8")

    @property
    def tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    def reset(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def close(self) -> None:
        """Close the log file if it's open."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        super().close()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit the log record. To be used by the logging module."""
        try:
            # Use the StructuredMessage if the message is an instance of it
            if isinstance(record.msg, LLMCallEvent):
                event = record.msg
                self._prompt_tokens += event.prompt_tokens
                self._completion_tokens += event.completion_tokens

                # Log to file if file handle is open
                if self._log_file:
                    self._log_file.write(str(event) + "\n")
                    self._log_file.flush()  # Ensure immediate write
        except Exception:
            self.handleError(record)


class OrderedMessageFilterAgent(MessageFilterAgent):
    """
    A MessageFilterAgent that maintains the original message order after filtering.

    Unlike the base MessageFilterAgent which groups messages by source,
    this implementation preserves the chronological order of messages
    while still applying the per-source filters.
    """

    def _apply_filter(self, messages: Sequence[BaseChatMessage]) -> Sequence[BaseChatMessage]:
        # Create a mapping of what messages to keep for each source
        messages_to_keep = set()

        for source_filter in self._filter.per_source:
            # Get messages from this source
            source_messages = [m for m in messages if m.source == source_filter.source]

            # Apply position and count filters
            if source_filter.position == "first" and source_filter.count:
                selected = source_messages[:source_filter.count]
            elif source_filter.position == "last" and source_filter.count:
                selected = source_messages[-source_filter.count:]
            else:
                # If position is None or count is None, include all messages from this source
                selected = source_messages

            # Add selected messages to our set
            messages_to_keep.update(selected)

        # Return messages in their original order, but only those that should be kept
        return [msg for msg in messages if msg in messages_to_keep]