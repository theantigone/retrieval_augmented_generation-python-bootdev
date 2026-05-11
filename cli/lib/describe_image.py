import mimetypes
import os
from google.genai import types
from lib.llm import MODEL, CLIENT, SAFETY_SETTINGS
from lib.search_utils import PROMPT_PATH


def describe_image(image: str, query: str) -> None:
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"
    with open(image, "rb") as f:
        img = f.read()
    describe_image_path = os.path.join(PROMPT_PATH, "describe_image.md")
    with open(describe_image_path, "r") as f:
        system_prompt = f.read()
    parts = (
        [
            system_prompt,
            types.Part.from_bytes(data=img, mime_type=mime),
            query.strip(),
        ],
    )
    print("Starting response...")
    response = CLIENT.models.generate_content(
        model=MODEL,
        contents=parts,
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
    )
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
