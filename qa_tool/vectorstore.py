from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVector, SparseVectorParams, VectorParams

from FlagEmbedding import BGEM3FlagModel

from typing import List
from uuid import uuid4
import re
import os


class BGEDenseEmbeddings(Embeddings):
    """LangChain-compatible dense embedder for BGE-M3 via FlagEmbedding."""
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", cache_dir: str = "embeddings"):
        self.model = BGEM3FlagModel(model_name, cache_dir=cache_dir, use_fp16=True)
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return out['dense_vecs']  # List[List[float]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class BGESparseEmbeddings(SparseEmbeddings):
    """LangChain-compatible sparse embedder for BGE-M3 via FlagEmbedding."""
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", cache_dir: str = "embeddings"):
        self.model = BGEM3FlagModel(model_name, cache_dir=cache_dir, use_fp16=True)
        self.device = device

    def _to_sparse_vector(self, sparse_data: dict[int, float]) -> SparseVector:
        indices, values = [], []
        for token_id, weight in sparse_data.items():
            if weight > 0:
                indices.append(int(token_id))
                values.append(float(weight))
        return SparseVector(indices=indices, values=values)

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        out = self.model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )
        return [self._to_sparse_vector(d) for d in out["lexical_weights"]]  # List[dict[int,float]]

    def embed_query(self, text: str) -> SparseVector:
        return self.embed_documents([text])[0]


def InitVectorStore(retrieval_mode="hybrid"):

    # init qdrant client
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection_name = "feedbacks_and_bugs"
    create_new_collection = False
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense_vector": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse_vector": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        create_new_collection = True
    except:
        client.collection_exists(collection_name=collection_name)

    # init vectorstore
    dense_embeddings  = BGEDenseEmbeddings()
    sparse_embeddings = BGESparseEmbeddings()
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID if retrieval_mode == "hybrid" else RetrievalMode.DENSE,
        vector_name="dense_vector",
        sparse_vector_name="sparse_vector",
    )

    # init docs if a new collection was created
    if create_new_collection:

        feedbacks_file = "ai_test_user_feedback.txt"
        with open(feedbacks_file, 'r', encoding='utf-8') as file:
            feedbacks = file.readlines()[1:]
        feedbacks = [re.sub(r"^Feedback #\d+: ", "", feedback).strip() for feedback in feedbacks]
        feedbacks = [Document(page_content=feedback, metadata={"source": "feedbacks"}) for feedback in feedbacks]

        bugs_file = "ai_test_bug_report.txt"
        with open(bugs_file, 'r', encoding='utf-8') as file:
            bugs = file.read().split("\n\n\n\n\n")
        bugs = ["\n".join(bug.split("\n")[1:]) for bug in bugs]
        bugs = [Document(page_content=bug, metadata={"source": "bugs"}) for bug in bugs]

        docs = feedbacks + bugs
        uuids = [str(uuid4()) for _ in range(len(docs))]
        qdrant.add_documents(documents=docs, ids=uuids)

    return qdrant

