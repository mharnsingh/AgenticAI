from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough


def build_qa_prompt(query_text, doc_filters, docs):
    """Build a QA prompt with a clear header and document content."""
    prompt_parts = []
    
    # Header with the user's query
    prompt_parts.append(f"User query: '{query_text}'\n")
    
    # Brief explanation based on active filters
    explanations = []
    if doc_filters.get("feedbacks"):
        explanations.append("customer feedback and experiences")
    if doc_filters.get("bugs"):
        explanations.append("bug reports and proposed fixes")
    
    if explanations:
        categories = " and ".join(explanations)
        prompt_parts.append(f"Here are the relevant documents about {categories}.\n")
    else:
        prompt_parts.append("Here are the relevant documents.\n")
    
    # List the retrieved documents using their page_content
    if docs:
        for i, doc in enumerate(docs, 1):
            prompt_parts.append(f"Doc {i}:\n{doc.page_content}\n")
    
    # Final instruction part
    prompt_parts.append("Using the above context, please provide a clear and concise answer to the user's query.")
    
    return "\n".join(prompt_parts)


def InitQAChain(retriever, answer_gen_llm):

    retriever_chain = RunnableParallel(
        query_text=RunnablePassthrough(),
        retriever_output=RunnableLambda(
            lambda query_text: retriever._get_relevant_documents(query_text["query"])
        ),
    )
    
    build_prompt = RunnableLambda(
        lambda inputs: build_qa_prompt(
            inputs["query_text"]["query"],   # user query
            inputs["retriever_output"][0],   # doc_filters
            inputs["retriever_output"][2],   # docs
        )
    )

    postprocess_response = RunnableLambda(
        lambda llm_output: {"answer": llm_output.content}
    )

    qa_chain = retriever_chain | build_prompt | answer_gen_llm | postprocess_response
    return qa_chain

