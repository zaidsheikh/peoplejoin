import sys
import json
import yaml
from yaml.representer import SafeRepresenter

# Custom string class to force literal block style (|)
class LiteralString(str):
    pass

# Register a custom representer for LiteralString
def literal_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

yaml.add_representer(LiteralString, literal_str_representer, Dumper=yaml.SafeDumper)

def process_message_content(obj):
    """If obj has a messages[] list, convert content fields with newlines."""
    if isinstance(obj, dict) and 'messages' in obj and isinstance(obj['messages'], list):
        for message in obj['messages']:
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                if isinstance(content, str) and '\n' in content:
                    message['content'] = LiteralString(content)
    return obj

def main():
    processed_items = []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            processed_obj = process_message_content(obj)
            processed_items.append(processed_obj)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}", file=sys.stderr)

    # Output the entire list as YAML to stdout
    yaml.dump(
        processed_items,
        sys.stdout,
        Dumper=yaml.SafeDumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000  # Prevent auto line wrapping
    )

if __name__ == "__main__":
    main()
