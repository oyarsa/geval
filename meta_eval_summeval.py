import argparse
from collections import defaultdict
import json
import re
from typing import Any

from prettytable import PrettyTable
from scipy.stats import kendalltau, pearsonr, spearmanr


def calculate_correlation(
    pred_score: list[float], human_score: list[float], result: dict[str, float]
) -> dict[str, float]:
    assert len(pred_score) == len(human_score)

    if not result:
        result = {"pearson": 0, "spearman": 0, "kendalltau": 0}

    result["pearson"] += pearsonr(pred_score, human_score)[0]
    result["spearman"] += spearmanr(pred_score, human_score)[0]
    result["kendalltau"] += kendalltau(pred_score, human_score)[0]

    return result


def print_correlations(result: dict[str, float], n: int) -> None:
    table = PrettyTable(["Pearson", "Spearman", "Kendall"])
    n = n or 1
    table.add_row(
        [
            round(result["pearson"] / n, 4),
            round(result["spearman"] / n, 4),
            round(result["kendalltau"] / n, 4),
        ]
    )
    print(table)


def parse_output(output: str) -> float:
    if matched := re.search(r"^ ?([\d\.]+)", output):
        try:
            score = float(matched[1])
        except Exception:
            score = 0
    else:
        score = 0
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fp", type=str, default="results/gpt4_rel_detailed.json"
    )
    parser.add_argument("--dimension", type=str, default="relevance")
    args = parser.parse_args()

    with open(args.input_fp) as f:
        jobj: list[dict[str, Any]] = json.load(f)

    pred_scores: dict[str, list[float]] = defaultdict(list)
    human_scores: dict[str, list[float]] = defaultdict(list)

    print("Calculating correlation for G-Eval")
    for item in jobj:
        doc_id = item["doc_id"]

        all_responses = item["all_responses"]
        all_scores = [parse_output(x) for x in all_responses]
        score = sum(all_scores) / len(all_scores)

        pred_scores[doc_id].append(score)
        human_scores[doc_id].append(item["scores"][args.dimension])

    print(f"{len(pred_scores)=}")
    print(f"{len(human_scores)=}")

    results = {"pearson": 0.0, "spearman": 0.0, "kendalltau": 0.0}
    d_ctr = 0
    for doc_id, pred_scores_doc in pred_scores.items():
        human_scores_doc = human_scores[doc_id]
        if len(set(human_scores_doc)) <= 1 or len(set(pred_scores_doc)) <= 1:
            continue

        results = calculate_correlation(pred_scores_doc, human_scores_doc, results)
        d_ctr += 1
    print_correlations(results, n=d_ctr)


if __name__ == "__main__":
    main()
