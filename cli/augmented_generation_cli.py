import argparse
from lib.rag import question_answering, doc_summarization, doc_citing


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="doc count to summarize",
    )

    rag_parser = subparsers.add_parser("citations", help="Perform RAG (search + cite)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="doc count to summarize",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            question_answering(query)
        case "summarize":
            doc_summarization(args.query, args.limit)
        case "citations":
            doc_citing(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
