from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch


class RouterSchema(BaseModel):
    """
    Determines which tool to invoke: QA or Summary, along with the reasoning behind the decision.
    """
    tool: Literal['qa', 'summary'] = Field(
        description="‘qa’ if the input is a standalone question to answer; ‘summary’ if the input is a structured bug/issue report to summarize."
    )
    reasoning: str = Field(
        description="A brief explanation of why the tool was chosen."
    )

system_prompt = '''
You will receive a user’s raw input. Decide whether it is:

  1. A standalone *question* that needs a document-search / QA answer (“qa”),  
  2. A *structured issue/bug report* that needs summarizing into key points (“summary”).

**Rules**  
• If the text is one or two sentences ending in a question mark, or asks “What…?”, “How…?”, etc., or asking the question, choose “qa”.  
• If the text contains *multiple fields* like “Title:”, “Description:”, “Steps to Reproduce:”, “Environment:”, “Severity:”, “Proposed Fix:”, or otherwise looks like a bug report template, choose “summary”.  

Provide your output as a JSON object with two fields: 'tool' and 'reasoning'. Ensure the 'reasoning' field explains why you selected the corresponding tool.
'''


def InitAgent(agent_llm, qa_chain, summary_chain):

    agent_fn = RunnableLambda(
        lambda state: agent_llm.with_structured_output(RouterSchema).invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state['query']),
            ]
        )
    )

    entry = RunnablePassthrough()
    agent_branch = RunnableBranch(
        (lambda state: agent_fn.invoke(state).tool == 'qa', qa_chain),
        summary_chain
    )

    agent = entry | agent_branch
    return agent