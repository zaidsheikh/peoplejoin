#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_model",
        type=str,
        default=os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B"),
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=os.environ.get("LLM_API_KEY", "no_api_key_provided"),
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default=os.environ.get("LLM_BASE_URL", "https://cmu.litellm.ai"),
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="output jsonl file (required)",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "peoplejoin-qa" / "test.jsonl",
        help="Path to the test file (default: data/peoplejoin-qa/test.jsonl)",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=580,
        help="Number of instances to process from the test file (default: 580)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "peoplejoin-qa",
        help="Path to the tenant data directory (default: peoplejoin/data/peoplejoin-qa)",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        help="Path to the log file (default: <output_file>.log)",
    )
    parser.add_argument("--enable-thinking", action="store_true", help="Enable LLM's thinking mode if supported")
    return parser.parse_args()


def main():
    args = get_args()
    if not args.log_file:
        args.log_file = args.output_file.with_suffix(".log")

    os.makedirs(args.output_file.parent, exist_ok=True)

    think_mode = "/think" if args.enable_thinking else "/no_think"

    print(f"Using LLM model: {args.llm_model}")
    print(f"LLM base URL: {args.llm_base_url}")
    print(f"Enable thinking: {args.enable_thinking}")
    client = OpenAI(
        base_url=f"{args.llm_base_url}",
        api_key=args.llm_api_key
    )

    test_data = Path(args.test_file).read_text().strip().split("\n")[:args.num_instances]

    with open(args.output_file, "a") as f_out, open(args.log_file, "a") as f_log:
        for line in tqdm(test_data):
            datum = json.loads(line)

            tqdm.write(f"{datum['datum_id']}: {datum['question']}")

            data = json.loads(Path(args.data_dir / f"{datum['tenant_id']}.json").read_text())
            all_docs = [
                doc
                for docs in data.get("user_id_to_documents", {}).values()
                for doc in docs
            ]

            query = (
                f"{json.dumps(all_docs, indent=2)}\n"
                "Based on the above data, answer the following question:\n"
                f"{datum['question']} {think_mode}"
            )

            try:
                completion = client.chat.completions.create(
                    model=args.llm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": query,
                        }
                    ],
                )
            except Exception as e:
                tqdm.write(f"ERROR (datum ID {datum['datum_id']}): {e}")
                f_log.write(f"ERROR (datum ID {datum['datum_id']}): {e}\n")
                continue

            response = completion.choices[0].message.content or ""
            reasoning, final_answer = "", response
            if "</think>" in response:
                reasoning, final_answer = response.split("</think>")
                reasoning = reasoning.replace("<think>", "").strip()

            output = {
                "datum_id": datum['datum_id'],
                "tenant_id": datum['tenant_id'],
                "question": datum['question'],
                "model": args.llm_model,
                "query": query,
                "response": response,
                "reasoning": reasoning.strip(),
                "final_answer": final_answer.strip(),
            }
            f_out.write(json.dumps(output) + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()
