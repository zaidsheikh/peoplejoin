import json
import os
import re
import sys
import traceback

from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory



def get_system_message() -> str:
    return "You are a helpful AI assistant."


def standardized_event_to_message(
    id,
    event: ApiAction | CodeAction | MessageAction | TextObservation | WebObservation,
) -> dict:
    if isinstance(event, TextObservation):
        if event.source == "user":
            event.source = "human"

        elif event.source == "agent":
            event.source = "gpt"

        elif event.source == "environment":
            event.source = "observation"

        else:
            raise ValueError(f"Wrong event source: {event.source}")
        return {"from": event.source, "value": event.content}

    else:
        raise ValueError(f"Unknown event type: {type(event)}\n{event}")


def process_row(line):
    trajectory = Trajectory(**json.loads(line))
    id = trajectory.id
    system_message = get_system_message()
    events = trajectory.content
    conversations = []
    for i in range(len(events)):
        event = events[i]
        if (
            event.name == "user"
            and not event.content.strip().startswith("You are")
            and not trajectory.details["agent_id"].startswith("orchestrator_")
        ):
            # Don't include the initial user message in non-orchestator-agent trajectories
            continue
        try:
            message = standardized_event_to_message(id, event)
            # HACK: ADP puts the dataset-specific system message as the first user message
            # Since we not using multiple datasets, use the first user message as system message
            if not conversations and message["value"].strip().startswith("You are"):
                system_message = message["value"]
                continue
            conversations.extend([message])
        except Exception as e:
            traceback.print_exc()
            print(e)
            return None
    return {
        "id": trajectory.id,
        "conversations": conversations,
        "system": system_message,
    }


def main():
    for line in sys.stdin:
        output_line = process_row(line)
        if output_line:
            print(json.dumps(output_line))


if __name__ == "__main__":
    main()
