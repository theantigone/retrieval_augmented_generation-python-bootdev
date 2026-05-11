import argparse
from lib.describe_image import describe_image


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_parser = subparsers.add_parser("image", help="Multimodal query rewriting")
    image_parser.add_argument("--image", type=str, help="Path to an image file")
    image_parser.add_argument(
        "--query", type=str, help="Text query to rewrite based on the image"
    )

    args = parser.parse_args()

    match args.command:
        case "image":
            describe_image(args.image, args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
