#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Summarize Geval scores from multiple JSON files into a single CSV."
    )
    parser.add_argument(
        "relevance_scores_file",
        type=Path,
        help="Path to the JSON file containing relevance scores.",
    )
    parser.add_argument(
        "consistency_scores_file",
        type=Path,
        help="Path to the JSON file containing consistency scores.",
    )
    parser.add_argument(
        "coherence_scores_file",
        type=Path,
        help="Path to the JSON file containing coherence scores.",
    )
    return parser.parse_args()


def compute_average_score(all_responses):
    scores = []
    for r in all_responses:
        try:
            scores.append(int(r[0]))
        except Exception:
            scores.append(1)
    return sum(scores) / len(scores)


if __name__ == "__main__":
    args = get_args()

    relevance_scores = json.loads(args.relevance_scores_file.read_text())
    consistency_scores = json.loads(args.consistency_scores_file.read_text())
    coherence_scores = json.loads(args.coherence_scores_file.read_text())

    summary = {}
    for item in relevance_scores:
        summary[item["doc_id"]] = {"relevance_score": compute_average_score(item["all_responses"])}

    for item in consistency_scores:
        summary[item["doc_id"]]["consistency_score"] = compute_average_score(item["all_responses"])

    for item in coherence_scores:
        summary[item["doc_id"]]["coherence_score"] = compute_average_score(item["all_responses"])

    print("doc_id,relevance_score,consistency_score,coherence_score")
    for doc_id, scores in summary.items():
        print(
            f"{doc_id},{scores['relevance_score']},{scores['consistency_score']},{scores['coherence_score']}"
        )

    metrics = ("relevance_score", "consistency_score", "coherence_score")
    avg = {m: sum(s[m] for s in summary.values()) / len(summary) for m in metrics}
    print(f"Average,{avg['relevance_score']},{avg['consistency_score']},{avg['coherence_score']}")