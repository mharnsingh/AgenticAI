from langchain_openai import ChatOpenAI
from qa_tool.vectorstore import InitVectorStore
from qa_tool.self_query_retreiver import InitSelfQueryRetreiver
from qa_tool.qa_chain import InitQAChain
from summary_tool.summary_chain import SummaryChainInit
from ai_agent.agent import InitAgent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import os


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "ai-agent-demo"


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, )

print("Initializing vectorstore...")
vectorstore = InitVectorStore(retrieval_mode="hybrid")

print("Initializing retriever...")
retriever = InitSelfQueryRetreiver(self_query_llm=llm, vectorstore=vectorstore, k_retriever=5)

print("Initializing agent...")
qa_chain = InitQAChain(retriever=retriever, answer_gen_llm=llm)
summary_chain = SummaryChainInit(summary_llm=llm)
agent = InitAgent(agent_llm=llm, qa_chain=qa_chain, summary_chain=summary_chain)


# Initialize FastAPI app
app = FastAPI()

class AgentRequest(BaseModel):
    query: str

@app.post("/query")
async def query_agent(request: AgentRequest):
    try:
        result = agent.invoke({"query": request.query})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
