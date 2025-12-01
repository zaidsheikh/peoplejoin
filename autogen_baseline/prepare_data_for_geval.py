#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir_or_file",
        type=Path,
        help="Directory containing *_autogen_messages.jsonl files from multi-agent peoplejoin-doc-creation implementation, or jsonl file containing generated messages from the single agent baseline",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to save the prepared data (json) for Geval",
    )
    parser.add_argument(
        "--reference_file",
        type=Path,
        help="Path to the reference jsonl file (default: peoplejoin/data/peoplejoin-doc-creation/test.scenario.jsonl)",
        default=Path(__file__).resolve().parent.parent / "data" / "peoplejoin-doc-creation" / "test.scenario.jsonl",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "peoplejoin-doc-creation" / "tenants",
        help="Path to the tenant data directory (default: peoplejoin/data/peoplejoin-doc-creation/tenants)",
    )
    return parser.parse_args()


def load_references(reference_file: Path) -> dict:
    references = {}
    with open(reference_file) as f:
        for line in f:
            data = json.loads(line)
            references[data["datum_id"]] = data
    return references


def load_hypotheses(input_dir_or_file: Path) -> dict:
    hypotheses = {}
    if input_dir_or_file.is_dir():
        for input_file in input_dir_or_file.glob("*_autogen_messages.jsonl"):
            datum_id = input_file.stem.replace("_autogen_messages", "")
            lines = input_file.read_text().strip().splitlines()
            for line in lines[::-1]:  # Reverse to get the last orchestrator message
                message_data = json.loads(line.strip())
                if message_data.get("source") == "orchestrator":
                    hypotheses[datum_id] = message_data.get("content", "").split("Final Summary:")[-1].strip()
                    break
    elif input_dir_or_file.is_file():
        for line in input_dir_or_file.read_text().splitlines():
            item = json.loads(line)
            final_answer = item["final_answer"].split("<summary>")[-1].split("</summary>")[0].strip()
            hypotheses[item["datum_id"]] = final_answer
    else:
        raise ValueError(f"Input path {input_dir_or_file} is neither a directory nor a file.")
    return hypotheses


def load_source_docs(data_dir: Path) -> dict:
    source_docs = {}
    for input_file in data_dir.glob("*.json"):
        data = json.loads(input_file.read_text())
        tenant_id = data["tenant_id"].split("/")[-1]
        source_docs[tenant_id] = {}
        for docsets in data["user_id_to_documents"].values():
            for doc in docsets:
                source_docs[tenant_id][doc["title"]] = doc["content"]
    return source_docs


if __name__ == "__main__":
    args = get_args()

    assert args.input_dir_or_file.exists(), f"Input path {args.input_dir_or_file} does not exist."
    assert args.reference_file.exists() and args.reference_file.is_file(), f"Reference file {args.reference_file} does not exist or is not a file."
    assert args.data_dir.exists() and args.data_dir.is_dir(), f"Data directory {args.data_dir} does not exist or is not a directory."

    print("Reference file:", args.reference_file)
    print("Input data:", args.input_dir_or_file)
    print("Output file:", args.output_file)

    references = load_references(args.reference_file)
    hypotheses = load_hypotheses(args.input_dir_or_file)
    all_source_docs = load_source_docs(args.data_dir)

    output_data = []
    for datum_id, ref_data in references.items():
        tenant_id = ref_data["tenant_id"].split("/")[-1]
        source_docs = [all_source_docs[tenant_id].get(title, "") for title in ref_data["gold_document_ids"]]
        output_data.append(
            {
                "doc_id": datum_id,
                "source": "\n".join(source_docs),
                "reference": ref_data["gold_summary"],
                "system_output": hypotheses.get(datum_id, ""),
            }
        )
    args.output_file.write_text(json.dumps(output_data, indent=2))


