import os
from dotenv import load_dotenv
from lib.search_utils import PROMPT_PATH
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# model = "gemma-3-27b-it"
model = "gemma-4-26b-a4b-it"
# model = "gemma-4-31b-it"
# model = "gemini-3.1-flash-lite-preview"
client = genai.Client(api_key=api_key)

MODEL = model


def generate_content(prompt: str, query: str) -> str:
    prompt = prompt.format(query=query)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        ),
    )
    #    response = client.models.generate_content(
    #        model="gemma-3-27b-it",
    #        contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    #    )
    #    print(response.text)
    #    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    #    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    return response.text


def augment_prompt(query: str, type: str) -> str:
    prompt_path = os.path.join(PROMPT_PATH, f"{type}.md")
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return generate_content(prompt, query)


if __name__ == "__main__":
    generate_content()
