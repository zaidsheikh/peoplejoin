#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing *_autogen_messages.jsonl files",
    )
    parser.add_argument(
        "--reference_file",
        type=Path,
        help="Path to the reference jsonl file",
        default=Path(__file__).parent.parent / "data" / "peoplejoin-doc-creation" / "test.scenario.jsonl",
    )
    return parser.parse_args()


def load_references(reference_file: Path) -> dict:
    references = {}
    with open(reference_file) as f:
        for line in f:
            data = json.loads(line)
            references[data["datum_id"]] = data["gold_summary"]
    return references


def load_hypotheses(input_dir: Path) -> dict:
    hypotheses = {}
    for input_file in input_dir.glob("*_autogen_messages.jsonl"):
        datum_id = input_file.stem.replace("_autogen_messages", "")
        lines = input_file.read_text().strip().splitlines()
        for line in lines[::-1]:  # Reverse to get the last orchestrator message
            message_data = json.loads(line.strip())
            if message_data.get("source") == "orchestrator":
                hypotheses[datum_id] = message_data.get("content", "").split("Final Summary:")[-1].strip()
                break
    return hypotheses


if __name__ == "__main__":
    args = get_args()

    print("Reference file:", args.reference_file)
    print("Input directory:", args.input_dir)

    references = load_references(args.reference_file)
    hypotheses = load_hypotheses(args.input_dir)

    all_references, all_hypotheses = [], []
    for datum_id, reference in references.items():
        hypothesis = hypotheses.get(datum_id, "")
        all_references.append(reference.strip().replace("\n", " "))
        all_hypotheses.append(hypothesis.strip().replace("\n", " "))

    print(scorer.score("\n".join(all_references), "\n".join(all_hypotheses)))
