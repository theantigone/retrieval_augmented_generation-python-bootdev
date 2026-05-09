import argparse
from lib.augmented_generation import augmented_generation


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            results, response = augmented_generation(query)
            print("Search Results:")
            for r in results:
                print("-", r["title"])
            print("\nRAG Response:")
            print(response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
