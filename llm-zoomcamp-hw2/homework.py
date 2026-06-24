from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
import numpy as np

from embedder import Embedder


QUERY_Q1 = "How does approximate nearest neighbor search work?"
QUERY_Q4 = "What metric do we use to evaluate a search engine?"
QUERY_Q5 = "How do I store vectors in PostgreSQL?"
QUERY_Q6 = "How do I give the model access to tools?"
PINNED_COMMIT = "8c1834d"


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


def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id=PINNED_COMMIT,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    return [file.parse() for file in reader.read()]


def build_indices():
    documents = load_documents()
    chunks = chunk_documents(documents, size=2000, step=1000)

    embedder = Embedder()
    vectors = embedder.encode_batch([chunk["content"] for chunk in chunks])

    vector_search = VectorSearch(keyword_fields=["filename"])
    vector_search.fit(vectors=vectors, payload=chunks)

    text_search = Index(text_fields=["content"], keyword_fields=["filename"])
    text_search.fit(chunks)

    return documents, chunks, embedder, vectors, vector_search, text_search


def question_1(embedder):
    v = embedder.encode(QUERY_Q1)
    return float(v[0]), v


def question_2(embedder, query_vector, documents):
    target = next(
        doc
        for doc in documents
        if doc["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md"
    )
    page_vector = embedder.encode(target["content"])
    return float(page_vector.dot(query_vector))


def question_3(chunks, chunk_vectors, query_vector):
    scores = chunk_vectors.dot(query_vector)
    best_idx = int(np.argmax(scores))
    return chunks[best_idx]["filename"], float(scores[best_idx])


def question_4(embedder, vector_search):
    query_vector = embedder.encode(QUERY_Q4)
    results = vector_search.search(query_vector=query_vector, num_results=5)
    return results[0]["filename"], results


def question_5(embedder, vector_search, text_search):
    query_vector = embedder.encode(QUERY_Q5)
    vector_results = vector_search.search(query_vector=query_vector, num_results=5)
    text_results = text_search.search(QUERY_Q5, num_results=5)

    vector_files = {result["filename"] for result in vector_results}
    text_files = {result["filename"] for result in text_results}
    return sorted(vector_files - text_files), vector_results, text_results


def question_6(embedder, vector_search, text_search):
    query_vector = embedder.encode(QUERY_Q6)
    vector_results = vector_search.search(query_vector=query_vector, num_results=5)
    text_results = text_search.search(QUERY_Q6, num_results=5)
    fused_results = rrf([vector_results, text_results])
    return fused_results[0]["filename"], fused_results


def main():
    documents, chunks, embedder, vectors, vector_search, text_search = build_indices()

    q1, query_vector = question_1(embedder)
    q2 = question_2(embedder, query_vector, documents)
    q3_file, q3_score = question_3(chunks, vectors, query_vector)
    q4_file, _ = question_4(embedder, vector_search)
    q5_only_vector, _, _ = question_5(embedder, vector_search, text_search)
    q6_file, _ = question_6(embedder, vector_search, text_search)

    print(f"Q1: {q1}")
    print(f"Q2: {q2}")
    print(f"Q3: {q3_file} ({q3_score})")
    print(f"Q4: {q4_file}")
    print(f"Q5: {q5_only_vector}")
    print(f"Q6: {q6_file}")


if __name__ == "__main__":
    main()
