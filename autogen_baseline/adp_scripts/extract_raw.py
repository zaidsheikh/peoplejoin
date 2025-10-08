import glob
import json
import os
import sys
from pathlib import Path

if not len(sys.argv) == 2:
    print(f"Usage: {sys.argv[0]} <input_dir>")
    sys.exit(1)

input_dir = sys.argv[1]
if not os.path.isdir(input_dir):
    print(f"Error: {input_dir} is not a directory")
    sys.exit(1)

for file_path in glob.glob(os.path.join(input_dir, "*_llm_calls.jsonl")):
    # the last llm call by orchestrator should have the full conversation history
    for line in reversed(Path(file_path).read_text().splitlines()):
        llm_call = json.loads(line)
        if llm_call["agent_id"].startswith("orchestrator_"):
            print(json.dumps(llm_call))
            break