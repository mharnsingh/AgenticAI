# AI Agent Demo

This repository contains an AI agent that routes user queries to two tools for question answering and summarization. The project demonstrates the following components:

- **ai_agent:**

  - Routes incoming queries to the appropriate tool (QA or Summary).

- **qa_tool:**

  - Uses the `baai/bge-m3` embedding model that supports both dense and sparse vector representations.
  - Leverages Qdrant as the vector store and performs hybrid search using both dense and sparse retrieval methods.
  - Implements a self-query retriever that automatically determines if the query is related to user feedback or bug reports.
  - Integrates a question answering chain that processes the user query along with the retrieved documents to generate an answer.

- **summary_tool:**

  - Implements a chain for generating structured summaries from issue or bug reports. This chain uses controlled output formatting to ensure the summary adheres to a predefined JSON schema.

- **demo_web_page:**
  - A Streamlit-based web interface is available for testing the AI agent via a web page at `http://localhost`.

All components are containerized using Docker and deployed via Docker Compose. An Nginx reverse proxy is configured to expose only external port 80 for improved deployment security. The application is also instrumented with LangSmith for monitoring (traces available at [smith.langchain.com](https://smith.langchain.com)).

## Setup and Installation

### 1. Clone the Repository

```sh
git clone https://github.com/mharnsingh/AgenticAI.git
cd AgenticAI
```

### 2. Configure Environment Variables

Create a `.env` file in the repository root with the following keys:

```
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

### 3. Run the Application with Docker Compose

Build and run all services by executing:

```sh
docker-compose up --build
```

This will start the following services:

- **agent:** FastAPI service handling incoming queries.
- **qdrant:** Vector store service for document retrieval.
- **streamlit:** Web interface for demonstrating the AI agent.
- **nginx:** Reverse proxy exposing the application on port 80.

## Usage

- **Web Demo:**  
  Open your browser and navigate to `http://localhost` to access the Streamlit demo interface.

- **API Endpoint:**  
  Send a POST request to the AI Agent API at `http://localhost/agent/query` with a JSON payload:

  ```json
  {
    "query": "your query here"
  }
  ```

  The agent routes the query to either the QA tool or the summary tool based on its content.

- **Web Demo:**  
   Interactions and traces are monitored via LangSmith. You can view traces at [smith.langchain.com](https://smith.langchain.com).
