import json
import sys

from schema.action.action import Action
from schema.observation.observation import Observation
from schema.observation.text import TextObservation
from schema.trajectory import Trajectory


def convert_step(step: dict[str, str]) -> list[Action | Observation]:
    if step["role"] == "system":
        return [
            TextObservation(
                content=step["content"],
                source="user",
                name="user",
            )
        ]
    else:
        return [
            TextObservation(
                content=step["content"],
                source="agent" if step["role"] == "assistant" else step["role"],
                name=step.get("name"),
            )
        ]


for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["messages"]:
        content.extend(convert_step(step))

    standardized_data = Trajectory(id=raw_data["datum_id"], content=content)
    print(standardized_data.model_dump_json())
