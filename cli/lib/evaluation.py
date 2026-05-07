import json
from .hybrid_search import HybridSearch
from .search_utils import GOLDEN_DATASET_PATH, load_movies


def load_test_cases():
    with open(GOLDEN_DATASET_PATH, "r") as f:
        test_cases = json.load(f)["test_cases"]
    return test_cases


def evaluate(limit):
    print(f"k={limit}\n")
    test_cases = load_test_cases()
    movies = load_movies()
    hs = HybridSearch(movies)
    # print(golden_dataset)
    for test_case in test_cases[:limit]:
        # print(test_case)
        # print(test_case["query"])
        # print(test_case["relevant_docs"])
        qry = test_case["query"]
        exp = test_case["relevant_docs"]
        rrf_results = hs.rrf_search(qry, 60, limit)
        relevant_cnt = 0
        for rrf_result in rrf_results:
            relevant_cnt += rrf_result["title"] in exp
        precision = relevant_cnt / len(rrf_results)
        recall = relevant_cnt / len(exp)
        f1 = 2 * (precision * recall) / (precision + recall)
        retrieved = ", ".join([r["title"] for r in rrf_results])
        print(f"- Query: {qry}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {retrieved}")
        print(f"  - Relevant: {', '.join(exp)}\n")
