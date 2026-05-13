import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from lib.search_utils import PROMPT_PATH
from sentence_transformers import CrossEncoder

from .llm import MODEL, SAFETY_SETTINGS

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

model = MODEL
client = genai.Client(api_key=api_key)


def individual_rerank(query: str, documents: list[dict]) -> list[dict]:
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
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
        )
        clean_response_text = (response.text or "").strip()
        try:
            clean_response_text = int(clean_response_text)
        except Exception as e:
            print(f"Failed to case {response.text} to int for {doc['title']}")
            print(e)
            clean_response_text = 0
        results.append({**doc, "rerank_individual": clean_response_text})
        print(f"Doc {idx} processed")
        time.sleep(3)
    results = sorted(results, key=lambda x: x["rerank_individual"], reverse=True)
    return results


def batch_rerank(query: str, documents: list[dict]) -> list[dict]:
    batch_rerank_path = os.path.join(PROMPT_PATH, "batch_rerank.md")
    with open(batch_rerank_path, "r") as f:
        prompt = f.read()
    _mtemp = """<movie id={idx}=>{title}:\n{desc}\n</movie>\n"""
    doc_list_str = ""
    for idx, doc in enumerate(documents):
        doc_list_str += _mtemp.format(
            idx=idx, title=doc["title"], desc=doc["description"]
        )
    print("Processing docs now...")
    _prompt = prompt.format(query=query, doc_list_str=doc_list_str)
    response = client.models.generate_content(
        model=model,
        contents=_prompt,
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
    )
    print("Finished processing docs")
    response_parsed = json.loads(response.text.strip("```json").strip("```").strip())
    results = []
    for idx, doc in enumerate(documents):
        results.append(
            {
                **doc,
                "rerank_batch": response_parsed.index(idx) + 1,
            }  # add 1 because the index starts at 0
        )  # `**doc` gets the value of each key in the for loop
    results = sorted(results, key=lambda x: x["rerank_batch"])
    return results


def cross_encoder_rerank(query: str, documents: list[dict]) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    # `predict` returns a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    results = []
    for score, doc in zip(scores, documents):
        results.append({**doc, "rerank_cross_encoding": score})
    results = sorted(results, key=lambda x: x["rerank_cross_encoding"], reverse=True)
    return results
