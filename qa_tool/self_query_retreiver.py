from pydantic import BaseModel, Field
from langchain_community.query_constructors.qdrant import QdrantTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from qdrant_client import models
from typing import List


class SelfQueryTags(BaseModel):
    """
    Categorizes the query topics and explains the decision.
    """
    feedbacks: bool = Field(
        description="True if the query is about user feedbacks."
    )
    bugs: bool = Field(
        description="True if the query is about bug or issue reports."
    )
    reason: str = Field(
        description="A brief explanation for the decision, indicating why the query was classified accordingly."
    )

system_prompt = '''
You are an assistant. Given a user query, decide if it is related to "user feedbacks", "bugs", or both.
Provide a brief explanation of your reasoning.
'''

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Query: {query}")
])


def self_query_parser_fn(response):
    try:
        doc_filters = {"feedbacks": response.feedbacks, "bugs": response.bugs}
        reasoning = response.reason
        return doc_filters, reasoning
    except Exception as e:
        print(f"Parsing text\n{response}\n raised following error:\n{e}")
        return True, True, "Error in parsing response"


class CustomSelfQueryRetriever(SelfQueryRetriever):

    k_retriever: int
    hybrid_fusion: models.FusionQuery

    def _get_relevant_documents(self, query: str) -> List[Document]:
        
        doc_filters, reason = self.query_constructor.invoke({"query": query})
        bugs, feedbacks = doc_filters["bugs"], doc_filters["feedbacks"]
        reason_str = f"Bugs: {bugs}, Feedbacks: {feedbacks}\nReasoning: {reason}"
        
        if not feedbacks and not bugs:
            feedbacks, bugs = True, True  # to search all documents

        # construct the search filter
        filters = []
        if feedbacks:
            filters.append(
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value="feedbacks")
                )
            )     
        if bugs:
            filters.append(
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value="bugs")
                )
            )

        search_filter = models.Filter(should=filters)
        search_kwargs = {
            "filter": search_filter, 
            "k": self.k_retriever,
            "hybrid_fusion": self.hybrid_fusion,
        }

        docs = self._get_docs_with_query(query, search_kwargs)
        return doc_filters, reason_str, docs


def InitSelfQueryRetreiver(
        self_query_llm, 
        vectorstore, 
        k_retriever=8, 
        hybrid_fusion="dbsf",
    ):

    query_constructor = (
        prompt_template | 
        self_query_llm.with_structured_output(SelfQueryTags) | 
        RunnableLambda(self_query_parser_fn)
    ).with_config({"run_name": "self-query retriever"})

    self_query_retriever = CustomSelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        structured_query_translator=QdrantTranslator(vectorstore.METADATA_KEY),
        k_retriever=k_retriever,
        hybrid_fusion=models.FusionQuery(fusion=models.Fusion.DBSF) if hybrid_fusion == "dbsf" else models.FusionQuery(fusion=models.Fusion.RRF),
        verbose=True,
    )
    
    return self_query_retriever

