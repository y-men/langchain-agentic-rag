import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",

]
docs = [ WebBaseLoader(url).load() for url in urls]
#doc_list = [ doc[0] for doc in docs]
doc_list = [ d for sublist in docs for d in sublist]

# This method creates a text splitter that's specifically designed
# to work with tiktoken encoding - which is the tokenization system used by
# OpenAI's models (like GPT-3.5 and GPT-4).

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0,
    # encoding_name="cl100k_base",
    # separators=["\n\n", "\n", " ", ""],
)
texts = text_splitter.split_documents(doc_list)

# The sentence-transformers/all-MiniLM-L6-v2 model is:
# Free to use - No API keys or rate limits
# Lightweight - Fast inference and small memory footprint
# High quality - Good performance for most embedding tasks
# Consistent - Same model you've used in other projects
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents( documents=texts,
                                     embedding=embeddings,
                                     collection_name="langchain_chroma_agentic_rag",
                                     persist_directory="./chroma_db",
                                     )
retriever = Chroma(
    collection_name="langchain_chroma_agentic_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
).as_retriever()
