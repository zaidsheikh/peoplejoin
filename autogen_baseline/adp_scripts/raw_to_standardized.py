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
    elif step["role"] == "user":
        if step.get("name") in ["user", "orchestrator"]:
            return [
                TextObservation(
                    content=step["content"],
                    source="user",
                    name=step.get("name"),
                )
            ]
        else:
            if step.get("name"):
                prefix = f"Response from @{step['name']}:"
                content = f"{prefix}\n{step['content']}" if not step["content"].startswith(prefix) else step["content"]
            else:
                content = step["content"]
            return [
                TextObservation(
                    content=content,
                    source="user",
                    name=step.get("name"),
                )
            ]
    elif step["role"] == "assistant":
        return [
            TextObservation(
                content=step["content"],
                source="agent",
                name=step.get("name", "orchestrator"),
            )
        ]
    else:
        raise ValueError(f"Unknown role: {step['role']}")


def remove_thinking_content(text: str) -> str:
    return text.split("</think>")[-1].strip()


for line in sys.stdin:
    raw_data = json.loads(line)

    content = []
    for step in raw_data["messages"]:
        content.extend(convert_step(step))

    final_message = raw_data["response"]["choices"][0]["message"]
    assert final_message["role"] == "assistant"

    if final_message["content"] is not None:
        final_response = remove_thinking_content(final_message["content"])

        content += [
            TextObservation(
                content=final_response,
                source="agent",
                name=None,
            )
        ]
    else:
        print(f"No final response in {raw_data["datum_id"]}", file=sys.stderr)

    standardized_data = Trajectory(
        id=raw_data["datum_id"],
        content=content,
        details={
            "datum_id": raw_data["datum_id"],
            "agent_id": raw_data["agent_id"],
        }
    )
    print(standardized_data.model_dump_json())
