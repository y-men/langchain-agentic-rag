import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field

""" 
This module contains the chains used in the graph
"""

load_dotenv()

class DocumentGrade(BaseModel):
    binary_grade: bool = Field(description="Document is relevant to the question, True or False")


# Anthropic Claude has excellent structured output support!
llm = ChatAnthropic(
    temperature=0,
    model="claude-3-haiku-20240307",  # Fast, affordable, great structured outputs
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
structured_llm = llm.with_structured_output(DocumentGrade)

#--- Grader Chain -----------------------------------------------

system_grader_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
Give a binary score True or False as a response to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader_prompt),
    ("human", "Retrieved Document: {document} \n Question: {question} \n Is the document relevant to the question?")
])

retrieved_doc_grader = grade_prompt | structured_llm

#--- Generator Chain -------------------------------------------------------------


prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, First try to review the context again, specially the last document. 
Then if you still don't know, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
""")
#hub.pull("rlm/rag-prompt")
generator_chain = prompt | llm | StrOutputParser()


# --- Halucination Chain -------------------------------------------------------------

from langchain.prompts import PromptTemplate

class HalucinationGrader(BaseModel):
    binary_grade: bool = Field(description="Answer grounded in the facts and documents in the context,\
     answer True or False")

haucination_system_prompt = """\
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'True' or 'False'. 'True' means that the answer is grounded in / supported by the set of facts.\
"""

haucination_prompt = ChatPromptTemplate.from_messages([
    ("system", haucination_system_prompt),
    ("human", "Set of falcts and documents: {documents}, LLM generation: {generation} \n \
                Is the generation grounded in the context?"),
])

halucianation_grader_llm = llm.with_structured_output(HalucinationGrader)
halucination_grader = haucination_prompt | halucianation_grader_llm

# --- Anser grader Chain -------------------------------------------------------------
class AnswerGrade(BaseModel):
    binary_grade: bool = Field(description="Answer addresses the question, True or False")

answer_system_prompt = """\
You are a grader assessing whether an LLM generation addresses the question. \n
Give a binary score 'True' or 'False'. 'True' means that the answer resolves the question.\
"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_system_prompt),
    ("human", "Question: {question}, LLM generation: {generation} \n \
                Does the generation address the question?"),
])

answer_grader_llm = llm.with_structured_output(AnswerGrade)
answer_grader = answer_prompt | answer_grader_llm

# --- Adaptive RAG: Router Chain -------------------------------------------------------------
from typing import Literal

class RouterQuery(BaseModel):
    target_datasource: Literal["websearch", "vectorstore"] = Field(
        ...,
        description="Give the user query choose between websearch or vectorstore",)

structured_router_llm = llm.with_structured_output(RouterQuery)
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """\
    You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. For all else, use web-search.    
    """
    ),
    ("human", "User query: {question}")])

router_chain = router_prompt | structured_router_llm













