#!/usr/bin/env python3

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_movies,
)

import argparse


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads properly")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Encode the text with embedding model"
    )
    embed_parser.add_argument("text", type=str, help="Text to be encoded")

    subparsers.add_parser(
        "verify_embeddings", help="Verify the embeddings loads properly"
    )

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Encode query with embedding model"
    )
    embedquery_parser.add_argument("query", type=str, help="User query to be encoded")

    search_parser = subparsers.add_parser("search", help="Search for relevant movies")
    search_parser.add_argument(
        "query", type=str, help="User query to search for relevant movies"
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Limit the number of movie results",
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_movies(args.query, args.limit)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
