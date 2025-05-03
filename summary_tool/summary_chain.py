from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


class IssueSummary(BaseModel):
    """
    Extracts key fields from a single issue report.
    """
    reported_issues: str = Field(
        description="A concise description of the problem reported."
    )
    affected_components: str = Field(
        description="Which feature(s) or component(s) are impacted by this issue."
    )
    severity: str = Field(
        description="The severity level (e.g., Low, Medium, High, Critical)."
    )


system_prompt = """\
You are an assistant that summarizes a single issue report.

Given the issue description below, extract and return a JSON object with exactly these fields:

- reported_issues: A concise statement of the reported problem.
- affected_components: The feature(s) or component(s) impacted.
- severity: The severity level (Low, Medium, High, Critical).

Respond _only_ with valid JSON conforming to the schema.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """\
Summarize the following issue report:
```
{query}
```
""")
])


def SummaryChainInit(summary_llm):
    postprocess_response = RunnableLambda(
        lambda llm_output: llm_output.dict()
    )
    chain = prompt_template | summary_llm.with_structured_output(IssueSummary) | postprocess_response
    return chain
