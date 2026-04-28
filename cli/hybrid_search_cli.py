import argparse

from lib.hybrid_search import normalize_scores, weighted_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="A hybrid search with a weighted average combination",
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="User query to find relative documents for"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="Percent of weight for BM25",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
