import argparse
import json
import sys
import traceback
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = """
You are a fair judge assistant tasked with providing clear, objective feedback and a score based on how well the response to a question aligns with the reference answer.

## Scoring Scale
0. The response does not align with the reference answer at all.
1. The response partially aligns with the reference answer.
2. The response fully aligns with the reference answer

## Instructions
- Valid scores are 0, 1, or 2 only.
- Formatting of the response doesn't matter and shouldn't affect the score.
- Order in which information is presented doesn't matter unless the question specifically requires it.
- Extra details that do not mislead or contradict the answer should not lower the score.

Your output should strictly follow this format:
feedback: <Brief explanation of why you gave this score>
score: <integer number 0, 1 or 2>
"""

USER_PROMPT = """
### Question
{question}

## Reference Answer
{reference}

## Response
{response}
"""

openai_client = OpenAI()


def get_cache_key(question: str, reference: str, response: str) -> str:
    return f"{question} ||| {reference} ||| {response}"


def score_response(question: str, reference: str, response: str, llm_model: str = "gpt-5-nano", cache=None) -> tuple[int, str]:
    cache_key = get_cache_key(question, reference, response)
    if cache is not None and cache_key in cache:
        score = cache[cache_key]["score"]
        if score != -1:
            return score, cache[cache_key].get("judge_response", "")

    prompt = USER_PROMPT.format(reference=reference, response=response, question=question)
    try:
        completion = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT + prompt},
            ],
        )
        answer = completion.choices[0].message.content or ""
        for line in answer.splitlines():
            if line.lower().startswith("score:"):
                score_str = line.split("score:")[1].strip()
                score = int(score_str)
                if score in [0, 1, 2]:
                    if cache is not None:
                        cache[cache_key] = {"score": score, "judge_response": answer}
                    return score, answer
                else:
                    return -1, answer
        tqdm.write("Score not found in the response.")
        return -1, ""
    except Exception as e:
        tqdm.write(f"Error during scoring: {e}")
        traceback.print_exc()
    return -1, ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "peoplejoin-qa" / "test.jsonl",
    )
    parser.add_argument("--saved_outputs_file", type=Path, required=True)
    parser.add_argument("--metrics_file", type=Path, required=True, help="JSONL file to save the computed metrics, if file exists, will append to it, processed datum_id will be skipped")
    parser.add_argument("--llm_model", type=str, default="azure/gpt-4.1")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Using LLM model: {args.llm_model}", file=sys.stderr)


    ground_truth = {}
    for line in args.ground_truth.read_text().splitlines():
        datum = json.loads(line)
        ground_truth[datum["datum_id"]] = datum
        # TODO: check for datum_attributes.datum_is_unanswerable

    cache = {}
    processed_datum_ids = set()
    if args.metrics_file.exists():
        for line in args.metrics_file.read_text().splitlines():
            data = json.loads(line)
            processed_datum_ids.add(data["datum_id"])
            cache_key = get_cache_key(data["question"], data["reference"], data["response"])
            cache[cache_key] = {
                "score": data["score"],
                "judge_response": data.get("judge_response", ""),
            }

    with open(args.metrics_file, "a") as metrics_file:
        outputs = [json.loads(line) for line in args.saved_outputs_file.read_text().splitlines()]
        for output in tqdm(outputs):
            datum_id, response = output["datum_id"], output["final_answer"]
            if datum_id in processed_datum_ids:
                tqdm.write(f"Skipping already processed datum_id: {datum_id}")
                continue


            reference = ground_truth[datum_id].get("execution_result", "No reference answer")
            question = ground_truth[datum_id].get("question", "No question provided")
            score, judge_response = score_response(
                question=question,
                reference=reference,
                response=response,
                llm_model=args.llm_model,
                cache=cache
            )
            data = {
                "datum_id": datum_id,
                "question": question,
                "reference": reference,
                "response": response,
                "score": score,
                "judge_response": judge_response,
            }
            tqdm.write(f"Datum ID: {datum_id}, Score: {score}")
            metrics_file.write(json.dumps(data) + "\n")



if __name__ == "__main__":
    main()
