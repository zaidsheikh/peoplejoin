import json
import logging
import warnings
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

from autogen_agentchat.agents import MessageFilterAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_core import CancellationToken
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


async def capture_stream_and_write_to_file(stream, jsonl_file_path):
    """
    Captures all events/messages from the stream and writes them to a JSONL file in real-time.

    Args:
        stream: The async generator stream to be captured.
        jsonl_file_path: The path to the JSONL file to write the events/messages.

    Returns:
        A new async generator with the same events/messages.
    """
    async def wrapped_stream():
        with open(jsonl_file_path, "w") as jsonl_file:
            async for message in stream:
                if hasattr(message, "model_dump") and callable(message.model_dump):
                    jsonl_file.write(json.dumps(message.model_dump(), default=str) + "\n")
                jsonl_file.flush()  # Ensure the message is written immediately
                yield message  # Pass the message along to Console()

    return wrapped_stream()



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
            if self._log_file_path.exists():
                self._log_file_path.unlink()
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

    Filtering is configured using :class:`MessageFilterConfig`, which supports:
    - Filtering by message source (e.g., only messages from "user" or another agent)
    - Selecting the first N or last N messages from each source
    - If position is `None`, all messages from that source are included

    This agent is compatible with both direct message passing and team-based execution
    such as :class:`~autogen_agentchat.teams.GraphFlow`.

    Example:
        >>> agent_a = MessageFilterAgent(
        ...     name="A",
        ...     wrapped_agent=some_other_agent,
        ...     filter=MessageFilterConfig(
        ...         per_source=[
        ...             PerSourceFilter(source="user", position="first", count=1),
        ...             PerSourceFilter(source="B", position="last", count=2),
        ...         ]
        ...     ),
        ... )
    """

    def _apply_filter(self, messages: Sequence[BaseChatMessage]) -> Sequence[BaseChatMessage]:
        # Create a mapping of what message indices to keep
        indices_to_keep = set()

        for source_filter in self._filter.per_source:
            # Get indices of messages from this source
            source_indices = [i for i, m in enumerate(messages) if m.source == source_filter.source]

            # Apply position and count filters to get the indices we want
            if source_filter.position == "first" and source_filter.count:
                selected_indices = source_indices[:source_filter.count]
            elif source_filter.position == "last" and source_filter.count:
                selected_indices = source_indices[-source_filter.count:]
            else:
                # If position is None or count is None, include all messages from this source
                selected_indices = source_indices

            # Add selected indices to our set
            indices_to_keep.update(selected_indices)

        # Return messages in their original order, but only those whose indices should be kept
        return [messages[i] for i in sorted(indices_to_keep)]

    def _update_message_source(self, message: BaseChatMessage) -> BaseChatMessage:
        """Update the source of a message if it matches the wrapped agent's name."""
        if message.source == self._wrapped_agent.name:
            message.source = self.name
        return message

    def _update_response_sources(self, response: Response) -> Response:
        """Update sources in a Response object."""
        if response.chat_message:
            response.chat_message = self._update_message_source(response.chat_message)

        # updated_inner_messages = None
        # if response.inner_messages:
        #     updated_inner_messages = []
        #     for msg in response.inner_messages:
        #         if isinstance(msg, BaseChatMessage):
        #             updated_inner_messages.append(self._update_message_source(msg))
        #         else:
        #             # BaseAgentEvent or other types pass through unchanged
        #             updated_inner_messages.append(msg)
        #     response.inner_messages = updated_inner_messages

        return response

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        """Override to update message sources in the response."""
        filtered = self._apply_filter(messages)
        response = await self._wrapped_agent.on_messages(filtered, cancellation_token)
        return self._update_response_sources(response)

    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """Override to update message sources in streamed responses."""
        filtered = self._apply_filter(messages)
        async for item in self._wrapped_agent.on_messages_stream(filtered, cancellation_token):
            if isinstance(item, BaseChatMessage):
                yield self._update_message_source(item)
            elif isinstance(item, Response):
                yield self._update_response_sources(item)
            else:
                # BaseAgentEvent and other types pass through unchanged
                yield item