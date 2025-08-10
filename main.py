
import os

from dotenv import load_dotenv

load_dotenv()
from graph.nodes import  app
import pprint


if __name__ == "__main__":
    print("Hello from langchain-agentic-rag!")
    pprint.pprint(app.invoke(input={"question": "Tell me about agent memory?"}))


