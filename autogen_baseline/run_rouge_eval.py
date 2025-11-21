#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer, scoring

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


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
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path to save the ROUGE evaluation results",
        default=None,
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

    rouge_lsum_scores: dict[str, dict[str, float]] = {}
    for datum_id, reference in references.items():
        hypothesis = hypotheses.get(datum_id, "")
        score = scorer.score(reference, hypothesis)
        r_score = score["rougeL"]
        rouge_lsum_scores[datum_id] = {
            "precision": r_score.precision,
            "recall": r_score.recall,
            "fmeasure": r_score.fmeasure,
        }

    f_measures = [v["fmeasure"] for v in rouge_lsum_scores.values()]
    average_rouge_lsum = sum(f_measures) / len(f_measures) if f_measures else 0.0
    print(f"Average ROUGE-L F1 Score: {average_rouge_lsum:.4f}")

    total_lcs = 0
    total_pred_tokens = 0
    total_ref_tokens = 0
    for datum_id, ref in references.items():
        pred = hypotheses.get(datum_id, "")
        pred_tokens = scorer._tokenizer.tokenize(pred)
        ref_tokens = scorer._tokenizer.tokenize(ref)

        rouge_lsum_scores[datum_id]["len_pred"] = len(pred_tokens)
        rouge_lsum_scores[datum_id]["len_ref"] = len(ref_tokens)

        # handle empty cases exactly like RougeScorer.score does
        if not pred_tokens or not ref_tokens:
            # lcs_length is 0, but still count token lengths (or follow your policy)
            total_pred_tokens += len(pred_tokens)
            total_ref_tokens += len(ref_tokens)
            rouge_lsum_scores[datum_id]["lcs_length"] = 0
            continue

        # use the internal DP table to get the exact integer LCS length
        lcs_table = rouge_scorer._lcs_table(ref_tokens, pred_tokens)
        lcs_length = lcs_table[-1][-1]

        total_lcs += lcs_length
        total_pred_tokens += len(pred_tokens)
        total_ref_tokens += len(ref_tokens)

        rouge_lsum_scores[datum_id]["lcs_length"] = lcs_length


    # compute micro-precision/recall/f1
    precision_micro = total_lcs / total_pred_tokens if total_pred_tokens else 0.0
    recall_micro = total_lcs / total_ref_tokens if total_ref_tokens else 0.0
    f1_micro = scoring.fmeasure(precision_micro, recall_micro)

    print('micro P', precision_micro, 'micro R', recall_micro, 'micro F1', f1_micro)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as out_f:
            out_f.write(
                "datum_id\tprecision\trecall\tfmeasure\tlen_pred\tlen_ref\tlcs_length\n"
            )
            for datum_id, metrics in rouge_lsum_scores.items():
                out_f.write(
                    f"{datum_id}\t"
                    f"{metrics['precision']:.4f}\t"
                    f"{metrics['recall']:.4f}\t"
                    f"{metrics['fmeasure']:.4f}\t"
                    f"{metrics['len_pred']}\t"
                    f"{metrics['len_ref']}\t"
                    f"{metrics['lcs_length']}\n"
                )
        print(f"Wrote per-datum ROUGE-L scores to {args.output_file}")
