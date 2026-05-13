import argparse

from lib.rag import (
    doc_citations,
    doc_summarization,
    question_answering,
    retrieving_augmented_generation,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize)"
    )
    sum_parser.add_argument("query", type=str, help="Search query for RAG")
    sum_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="doc count to summarize",
    )

    cit_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + summarize + cite)"
    )
    cit_parser.add_argument("query", type=str, help="Search query for RAG")
    cit_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="doc count to summarize w/ citations",
    )

    question_parser = subparsers.add_parser(
        "question", help="Perform RAG (search + generate answer to question)"
    )
    question_parser.add_argument("question", type=str, help="Search query for RAG")
    question_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="doc count to summarize w/ citations",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            retrieving_augmented_generation(query)
        case "summarize":
            doc_summarization(args.query, args.limit)
        case "citations":
            doc_citations(args.query, args.limit)
        case "question":
            question_answering(args.question, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
