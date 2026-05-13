import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embeddings load"
    )
    verify_parser.add_argument(
        "image_fpath", type=str, help="Path to the image to process"
    )

    image_parser = subparsers.add_parser(
        "image_search", help="Search for image embeddings"
    )
    image_parser.add_argument(
        "image_fpath", type=str, help="Path to the image to process"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_fpath)
        case "image_search":
            image_search_command(args.image_fpath)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
