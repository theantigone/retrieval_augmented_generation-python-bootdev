import os
import json
from lib.search_utils import PROMPT_PATH
from dotenv import load_dotenv
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
SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
]


def generate_content(prompt: str, query: str, **kwargs) -> str | None:
    prompt = prompt.format(query=query, **kwargs)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
    )
    # response = client.models.generate_content(
    # model="gemma-3-27b-it",
    # contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    # )
    # print(response.text)
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    # print("Response:", response)
    # print("Response candidates:", response.candidates)
    # print("Response prompt feedback:", response.prompt_feedback)
    if not response or not response.text:
        raise RuntimeError(f"Empty response: {response}")
    return response.text


def augment_prompt(query: str, type: str) -> str | None:
    prompt_path = os.path.join(PROMPT_PATH, f"{type}.md")
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return generate_content(prompt, query)


def llm_judge(query: str, formatted_results: str) -> str:
    llm_judge_path = os.path.join(PROMPT_PATH, "llm_judge.md")
    with open(llm_judge_path, "r") as f:
        prompt = f.read()
    print("Evaluating docs now...")
    response = generate_content(prompt, query, formatted_results=formatted_results)
    if not response:
        raise RuntimeError(f"Empty response: {response}")
    try:
        evaluations = json.loads(response)
    except Exception as e:
        print("Failed Response:", response)
        raise e
    print("Successfully evaluated docs")
    return evaluations


def _rag(query: str, documents: list[dict], prompt_fname: str):
    prompt_fpath = os.path.join(PROMPT_PATH, prompt_fname)
    with open(prompt_fpath, "r") as f:
        prompt = f.read()
    print("\nStarting RAG process...")
    response = generate_content(prompt, query, docs=documents)
    if not response:
        raise RuntimeError("Empty response:", response)
    print("Successfully ran RAG process")
    return response


def answer_question(query: str, documents: list[dict]) -> str:
    return _rag(query, documents, "answer_question.md")


def summarize_documents(query: str, documents: list[dict]) -> str:
    return _rag(query, documents, "summarization.md")


if __name__ == "__main__":
    generate_content()
