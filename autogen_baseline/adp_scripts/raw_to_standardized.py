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


def remove_thinking_content(text: str) -> str:
    return text.split("</think>")[-1].strip()


for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["messages"]:
        content.extend(convert_step(step))

    final_message = raw_data["response"]["choices"][0]["message"]
    assert final_message["role"] == "assistant"

    final_response = remove_thinking_content(final_message["content"])

    content += [
        TextObservation(
            content=final_response,
            source="agent",
            name=None,
        )
    ]

    standardized_data = Trajectory(id=raw_data["datum_id"], content=content)
    print(standardized_data.model_dump_json())
