import argparse

from lib.hybrid_search import normalize_scores, weighted_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize BM25 scores into a `0-1` scale"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Search using keywords and semantics, weighted based on the alpha value",
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="User query to perform a weighted search on"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="Dynamically control the weighted between the two scores",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit the number of results to show",
    )

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            weighted_results = weighted_search(args.query, args.alpha, args.limit)
            for i, (idx, weighted_result) in enumerate(
                zip(weighted_results.items[0], weighted_results)
            ):
                print(f"{i+1}. {weighted_result[idx]['title']}")
                print(f"\tHybrid Score: {weighted_result['hybrid_score']}")
                print(
                    f"\tBM25: {weighted_result['bm25_score']}, Semantic: {weighted_result['semantic_score']}"
                )
                print(f"\t{weighted_score[idx]['description']}")
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
