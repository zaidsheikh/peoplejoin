import json
import sys

from schema.action.action import Action
from schema.action.message import MessageAction
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    if step["role"] == "user":
        # TODO: need to add "name" field to TextObservation
        return [TextObservation(content=f"{step['name']}: {step['content']}", source=step["role"])]
    elif step["role"] == "assistant":
        return [MessageAction(content=step["content"])]
    elif step["role"] == "system":
        return []
    else:
        raise ValueError(f"Unknown role: {step['role']}")


for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["messages"]:
        content.extend(convert_step(step))

    standardized_data = Trajectory(id=raw_data["datum_id"], content=content)
    print(standardized_data.model_dump_json())
