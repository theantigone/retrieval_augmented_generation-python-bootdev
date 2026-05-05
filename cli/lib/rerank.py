import os
import time
import json
from dotenv import load_dotenv
from lib.search_utils import PROMPT_PATH
from google import genai
from google.genai import types
from .llm import MODEL

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

model = MODEL
client = genai.Client(api_key=api_key)


def individual_rerank(query: str, documents: dict):
    individual_rerank_path = os.path.join(PROMPT_PATH, "individual_rerank.md")
    with open(individual_rerank_path, "r") as f:
        prompt = f.read()
    results = []
    for idx, doc in enumerate(documents, start=1):
        _prompt = prompt.format(
            query=query, title=doc["title"], description=doc["description"]
        )
        response = client.models.generate_content(
            model=model,
            contents=_prompt,
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ]
            ),
        )
        clean_response_text = (response.text or "").strip()
        try:
            clean_response_text = int(clean_response_text)
        except Exception as e:
            print(f"Failed to case {response.text} to int for {doc['title']}")
            print(e)
            clean_response_text = 0
        results.append({**doc, "rerank_response": clean_response_text})
        print(f"Doc {idx} processed")
        time.sleep(3)
    results = sorted(results, key=lambda x: x["rerank_response"], reverse=True)
    return results


def batch_rerank(query: str, documents: dict):
    print("Processing docs now...")
    individual_rerank_path = os.path.join(PROMPT_PATH, "batch_rerank.md")
    with open(individual_rerank_path, "r") as f:
        prompt = f.read()
    _mtemp = """<movie id={idx}>{title}:\n{desc}\n</movie>\n"""
    doc_list_str = ""
    for idx, doc in enumerate(documents):
        doc_list_str += _mtemp.format(
            idx=idx, title=doc["title"], desc=doc["description"]
        )
    _prompt = prompt.format(query=query, doc_list_str=doc_list_str)
    response = client.models.generate_content(
        model=model,
        contents=_prompt,
        config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        ),
    )
    response_parsed = json.loads(response.text)
    results = []
    for idx, doc in enumerate(documents):
        results.append({**doc, "rerank_score": response_parsed.index(idx)})
    results = sorted(results, key=lambda x: x["rerank_score"])
    return results
