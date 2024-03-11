import argparse
import json
import time
from typing import Any

import openai
import tqdm


def run_instance(
    model: str, prompt: str, instance: dict[str, Any]
) -> dict[str, Any] | None:
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=2,
                max_tokens=5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=20,
            )
            assert isinstance(response, dict)

            time.sleep(0.5)

            return instance | {
                "all_responses": [c["message"]["content"] for c in response["choices"]]
            }
        except Exception as e:
            print(e)
            if "limit" in str(e):
                time.sleep(2)
            else:
                return None


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--prompt_fp", type=str, default="prompts/summeval/con_detailed.txt"
    )
    argparser.add_argument(
        "--save_fp", type=str, default="results/gpt4_con_detailed_openai.json"
    )
    argparser.add_argument("--summeval_fp", type=str, default="data/summeval.json")
    argparser.add_argument("--key", type=str, required=True)
    argparser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
    # argparser.add_argument("--model", type=str, default="gpt-4-0613")
    args = argparser.parse_args()
    openai.api_key = args.key

    with open(args.summeval_fp) as f:
        summeval: list[dict[str, Any]] = json.load(f)
    with open(args.prompt_fp) as f:
        prompt = f.read()

    ignore = 0
    new_json: list[dict[str, Any]] = []

    for instance in tqdm.tqdm(summeval):
        source = instance["source"]
        system_output = instance["system_output"]
        cur_prompt = prompt.replace("{{Document}}", source).replace(
            "{{Summary}}", system_output
        )
        instance["prompt"] = cur_prompt

        if new_instance := run_instance(args.model, cur_prompt, instance):
            new_json.append(new_instance)
        else:
            ignore += 1
            print("ignored", ignore)

    print("ignored total", ignore)
    with open(args.save_fp, "w") as f:
        json.dump(new_json, f, indent=4)


if __name__ == "__main__":
    main()
