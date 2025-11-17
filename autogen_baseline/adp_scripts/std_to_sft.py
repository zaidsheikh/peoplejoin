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

dataset = os.getenv("MY_DATASET")
assert dataset, "Please set the environment variable MY_DATASET"


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
    std_dataset = [json.loads(line)]
    std_data = std_dataset[0]
    trajectory = Trajectory(**std_data)
    id = trajectory.id
    system_message = get_system_message()
    events = trajectory.content
    conversations = []
    for i in range(len(events)):
        event = events[i]
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
    count = 0
    for line in sys.stdin:
        output_line = process_row(line)
        if output_line:
            with open(f"datasets/{dataset}/full_sft.jsonl", "a") as f:
                try:
                    f.write(json.dumps(output_line) + "\n")
                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    continue
        count += 1
        if count % 10000 == 0:
            print(count, file=sys.stderr)


if __name__ == "__main__":
    main()
