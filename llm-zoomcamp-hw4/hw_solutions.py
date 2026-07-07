import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from minsearch import Index, VectorSearch
from openai import OpenAI
from pydantic import BaseModel

from gitsource import GithubRepositoryDataReader, chunk_documents


ROOT_DIR = Path(__file__).resolve().parent.parent
HW2_DIR = ROOT_DIR / "llm-zoomcamp-hw2"
if str(HW2_DIR) not in sys.path:
    sys.path.append(str(HW2_DIR))

from embedder import Embedder


load_dotenv()

PINNED_COMMIT = "8c1834d"
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROUND_TRUTH_PATH = ROOT_DIR / "llm-zoomcamp-hw4" / "ground-truth.csv"
EMBEDDER_MODEL_PATH = ROOT_DIR / "llm-zoomcamp-hw2" / "models" / "Xenova" / "all-MiniLM-L6-v2"


class Questions(BaseModel):
    questions: list[str]


def llm_structured_groq(client, instructions, user_prompt, output_type, model):
    json_instructions = """
Return only valid JSON.
Use exactly this schema:
{
  "questions": ["question 1", "question 2", "question 3", "question 4", "question 5"]
}
""".strip()

    messages = [
        {"role": "system", "content": f"{instructions}\n\n{json_instructions}"},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    parsed_json = json.loads(content)
    parsed = output_type.model_validate(parsed_json)

    return parsed, response.usage


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()


def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id=PINNED_COMMIT,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    return [file.parse() for file in reader.read()]


def build_question_generation_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env")

    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )


def generate_ground_truth_sample(documents):
    target_files = {
        "01-agentic-rag/lessons/01-intro.md",
        "01-agentic-rag/lessons/02-environment.md",
        "01-agentic-rag/lessons/03-rag.md",
    }
    docs_3 = [doc for doc in documents if doc["filename"] in target_files]

    client = build_question_generation_client()
    ground_truth = []
    input_tokens = []

    for doc in docs_3:
        user_prompt = json.dumps(
            {
                "filename": doc["filename"],
                "content": doc["content"],
            },
            ensure_ascii=False,
        )

        parsed, usage = llm_structured_groq(
            client=client,
            instructions=data_gen_instructions,
            user_prompt=user_prompt,
            output_type=Questions,
            model=MODEL_NAME,
        )

        input_tokens.append(usage.prompt_tokens)

        for question in parsed.questions:
            ground_truth.append(
                {
                    "question": question,
                    "filename": doc["filename"],
                }
            )

    avg_input_tokens = sum(input_tokens) / len(input_tokens)
    return ground_truth, input_tokens, avg_input_tokens


def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH_PATH)
    return df.to_dict(orient="records")


def build_search_objects(documents):
    chunks = chunk_documents(documents, size=2000, step=1000)

    embedder = Embedder(path=EMBEDDER_MODEL_PATH)
    vectors = embedder.encode_batch([chunk["content"] for chunk in chunks])

    text_index = Index(text_fields=["content"], keyword_fields=["filename"])
    text_index.fit(chunks)

    vector_index = VectorSearch(keyword_fields=["filename"])
    vector_index.fit(vectors=vectors, payload=chunks)

    return chunks, embedder, text_index, vector_index


def build_search_functions(embedder, text_index, vector_index):
    def text_search(query, num_results=5):
        return text_index.search(query, num_results=num_results)

    def vector_search(query, num_results=5):
        query_vector = embedder.encode(query)
        return vector_index.search(query_vector=query_vector, num_results=num_results)

    def hybrid_search(query, k=60):
        text_results = text_search(query, num_results=10)
        vector_results = vector_search(query, num_results=10)
        return rrf([text_results, vector_results], k=k)

    return text_search, vector_search, hybrid_search


def compute_relevance(record, search_function):
    question = record["question"]
    expected_filename = record["filename"]
    results = search_function(question)
    return [1 if doc["filename"] == expected_filename else 0 for doc in results]


def hit_rate(relevance_scores):
    return sum(1 if any(scores) else 0 for scores in relevance_scores) / len(relevance_scores)


def mrr(relevance_scores):
    total = 0.0

    for scores in relevance_scores:
        for rank, value in enumerate(scores, start=1):
            if value == 1:
                total += 1 / rank
                break

    return total / len(relevance_scores)


def evaluate(ground_truth, search_function):
    relevance_scores = [
        compute_relevance(record, search_function)
        for record in ground_truth
    ]
    return {
        "hit_rate": hit_rate(relevance_scores),
        "mrr": mrr(relevance_scores),
    }


def main():
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")

    sample_ground_truth, input_tokens, avg_input_tokens = generate_ground_truth_sample(documents)
    print(f"Q1 input tokens: {input_tokens}")
    print(f"Q1 average input tokens: {avg_input_tokens}")
    print(f"Sample generated records: {sample_ground_truth[:3]}")

    if GROUND_TRUTH_PATH.exists():
        ground_truth = load_ground_truth()
        print(f"Loaded full ground truth rows: {len(ground_truth)}")
    else:
        ground_truth = None
        print(f"ground-truth.csv not found at {GROUND_TRUTH_PATH}")

    chunks, embedder, text_index, vector_index = build_search_objects(documents)
    text_search, vector_search, hybrid_search = build_search_functions(
        embedder,
        text_index,
        vector_index,
    )

    print(f"Chunks generated: {len(chunks)}")

    if ground_truth:
        query = ground_truth[0]["question"]
        print(f"First ground truth question: {query}")
        print(f"First text result: {text_search(query)[0]['filename']}")
        print(f"First vector result: {vector_search(query)[0]['filename']}")
        print(f"First hybrid result: {hybrid_search(query)[0]['filename']}")
        print(f"Text search metrics: {evaluate(ground_truth, text_search)}")
        print(f"Vector search metrics: {evaluate(ground_truth, vector_search)}")


if __name__ == "__main__":
    main()
