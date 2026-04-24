#!/usr/bin/env python3

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_movies,
    chunk_text,
    semantically_chunk_text,
    embed_chunks,
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

    chunk_parser = subparsers.add_parser(
        "chunk", help="Divides document into multiple chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Document to be chunked")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=200,
        help="Number of words in each fixed sized chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Extra words to give chunks context",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk based on semantic meaning"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Document to be chunked")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=4,
        help="Number of sentences in each fixed sized chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Extra sentences to give chunks context",
    )

    embed_chunk_parser = subparsers.add_parser(
        "embed_chunks", help="Chunk semantic embeddings"
    )

    args = parser.parse_args()

    match args.command:
        case "embed_chunks":
            embed_chunks()
        case "semantic_chunk":
            semantically_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
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
