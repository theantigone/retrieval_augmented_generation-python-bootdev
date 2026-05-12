import argparse
import mimetypes

from google.genai import types
from lib.llm import CLIENT, MODEL, SAFETY_SETTINGS

system_prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def describe_image(image: str, query: str) -> None:
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"
    with open(image, "rb") as f:
        img = f.read()
    parts = (
        [
            system_prompt,
            types.Part.from_bytes(data=img, mime_type=mime),
            query.strip(),
        ],
    )
    print("\nDescribing image...")
    response = CLIENT.models.generate_content(
        model=MODEL,
        contents=parts,
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
    )
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, help="Path to image for the rewriting")
    parser.add_argument("--query", type=str, help="User query that goes with the image")

    args = parser.parse_args()

    describe_image(args.image, args.query)


if __name__ == "__main__":
    main()
