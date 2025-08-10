"""
Run tests:
pytest . -s -v

"""
import os

from dotenv import load_dotenv

from graph.chains import DocumentGrade, retrieved_doc_grader
from injestion import retriever

load_dotenv()

def test_retrieved_doc_grader_yes():
    question = "What is an agent?"
    documents = retriever.invoke(question)
    result = retrieved_doc_grader.invoke({
        "document": documents[0],
        "question": question
    })
    

    # Check that we got a boolean response
    assert isinstance(result.binary_grade, bool)
    # For this test, we expect it to be relevant (True)
    assert result.binary_grade == True

def test_retrieved_doc_grader_no():
    question = "What is an agent?"
    documents = retriever.invoke(question)
    result = retrieved_doc_grader.invoke({
        "document": documents[0],
        "question": "How to make pizza ? "
    })

    assert isinstance(result.binary_grade, bool)
    assert result.binary_grade == False

from graph.chains import halucination_grader

def test_halucination_grader_yes():
    question = "What is an agent?"
    documents = retriever.invoke(question)
    generation = "Agents are a form of AI"
    result = halucination_grader.invoke({
        "generation": generation,
        "documents": documents
    })
    assert isinstance(result.binary_grade, bool)
    assert result.binary_grade == True

def test_halucination_grader_no():
    question = "What is an agent?"
    documents = retriever.invoke(question)
    generation = "We are preparing pizza with cheese"
    result = halucination_grader.invoke({
        "generation": generation,
        "documents": documents
    })
    assert isinstance(result.binary_grade, bool)
    assert result.binary_grade == False