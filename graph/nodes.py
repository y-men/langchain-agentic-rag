import pprint
from typing import Dict, List, Optional, TypedDict

from graph.chains import retrieved_doc_grader, generator_chain
from dotenv import load_dotenv
from langchain_core.documents import Document

from graph.state import GraphState
from injestion import retriever
from logger import logger

load_dotenv()


def retrieve_node(state: GraphState) -> Dict[str, any]:
    logger.info("RETRIEVING NODE")
    question = state["question"]
    logger.debug(f"{question=}")
    documents = retriever.invoke(question)
    return {"question": question, "documents": documents}




# ----------------------------------------------------------------

def document_grade_node(state: GraphState) -> Dict[str, any]:
    logger.info("DOCUMENT GRADE NODE")
    q = state["question"]
    docs = state["documents"]
    is_web_search = False
    #logger.debug(f"{q=} {docs=}")
    filtered_docs =[]
    for d in docs:
        result = retrieved_doc_grader.invoke({
            "question": q,
            "document": d,
        })
        logger.debug(f" Is the document relevant to the question? { result=}")
        if result.binary_grade:
            filtered_docs.append(d)
        else:
            is_web_search = True
            continue
    logger.debug(f"{is_web_search=}")
    return {"question": q, "documents": filtered_docs, "is_web_search": is_web_search}


# ----------------------------------------------------------------
from langchain_tavily import TavilySearch


def web_search_node(state: GraphState) -> Dict[str, any]:
    logger.info("WEB SEARCH NODE")
    question = state["question"]
    documents = state["documents"]

    tavily = TavilySearch( max_results=3)
    tavily_results = tavily.invoke({ "query": question})
    if tavily_results:
        results = "\n".join([d["content"] for d in tavily_results["results"]])
        web_doc = Document(
            page_content=results,
            metadata={
                "source": "web_search",
                "title": f"Web Search Results for: {question}",
                "description": f"Fresh web search results specifically retrieved to answer: {question}",
                "relevance": "high",
                "search_query": question,
                "document_type": "web_search_result"
            }
        )
      #  documents = [web_doc] + ( documents or [])
        (documents := documents or []).append(web_doc) #Try to put the search last (recency bias)
      #  documents = [web_doc]
    return {"question": question, "documents": documents}

# ----------------------------------------------------------------

def generate_node(state: GraphState) -> Dict[str, any]:
    logger.info("GENERATE NODE")
    question = state["question"]
    documents = state["documents"]
    generation = generator_chain.invoke({"question": question, "context": documents})
    return {"question": question, "generation": generation}


# --- Graph -------------------------------------------------------------

from langgraph.graph import StateGraph, END
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"

g = StateGraph(GraphState)
g.add_node(RETRIEVE, retrieve_node)
g.add_node(GRADE_DOCUMENTS, document_grade_node)
g.add_node(GENERATE, generate_node)
g.add_node(WEBSEARCH, web_search_node)

g.set_entry_point(RETRIEVE)
g.add_conditional_edges(
    GRADE_DOCUMENTS,
    lambda state: WEBSEARCH if state["is_web_search"] else GENERATE,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE
    }
)

g.add_edge(RETRIEVE, GRADE_DOCUMENTS)
g.add_edge(WEBSEARCH,GENERATE)
g.add_edge(GENERATE, END)

# --- A Self-RAG Loop: Hallucination and validation check edge --------------------------------------------

from graph.chains import halucination_grader, answer_grader


def validate_answer(state: GraphState) -> str:
    logger.info("Node validation:")
    ## TODO: Add aspect logging: before and after
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    halucination_result = halucination_grader.invoke({
        "documents": documents,
        "generation": generation
        })
    
    if halucination_result.binary_grade: # True == grounded (No hallucination)
        answer_result = answer_grader.invoke({
            "question": question,
            "generation": generation
            })
        logger.debug(f"Is the answer grounded in the documents? {halucination_result.binary_grade}")
        if answer_result.binary_grade: # True == True (Answer addresses the question)
            logger.debug(f"Is the answer  address the question? {answer_result.binary_grade} ")
            result = "valid"
        else:
            result = "invalid"
    else:
        result = "hallucinated"
    
    logger.info(f"validate_answer returning: {result}")
    return result

g.add_conditional_edges(GENERATE, validate_answer, {
    "invalid": GENERATE,
    "hallucinated": WEBSEARCH,
    "valid": END
})

# --- Adaptive RAG: Router Chain -------------------------------------------------------------
from graph.chains import router_chain
def router_direction(state: GraphState) -> str:
    question = state["question"]
    result = router_chain.invoke({"question": question})
    logger.debug(f"router_direction: {result.target_datasource=}")
    return result.target_datasource

g.set_conditional_entry_point(
    router_direction,
    {
        "websearch": WEBSEARCH,
        "vectorstore": RETRIEVE
    }
)



app = g.compile()



if __name__ == "__main__":
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
