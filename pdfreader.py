import os
import time
from getpass import getpass

import kdbai_client as kdbai
import pandas as pd
import requests
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import KDBAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

KDBAI_ENDPOINT = "your-endpoint"
KDBAI_API_KEY = "your-kdb-api-key"

OPENAI_API_KEY = "your-api-key"

TEMP = 0.0
K = 3
print("Create a KDB.AI session...")
session = kdbai.Session(endpoint=KDBAI_ENDPOINT, api_key=KDBAI_API_KEY)
session.list()

print('Create table "your-table"...')
schema = {
    "columns": [
        {"name": "id", "pytype": "str"},
        {"name": "text", "pytype": "bytes"},
        {
            "name": "embeddings",
            "pytype": "float32",
            "vectorIndex": {"dims": 1536, "metric": "L2", "type": "hnsw"},
        },
        {"name": "tag", "pytype": "str"},
        {"name": "title", "pytype": "bytes"},
    ]
}
# session.table("your-table").drop()
table = session.create_table("your-table", schema)
# table = session.table("your-table")  # Reuse the documents


print("Read a PDF...")
loader = PyPDFLoader("pdf-file-location.pdf")
pages = loader.load_and_split()
len(pages)


def add_texts_with_rate_limiting(
    vectordb, texts, metadatas, batch_size=5, sleep_time=60
):
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadata = metadatas.iloc[i : i + batch_size]
        vectordb.add_texts(texts=batch_texts, metadatas=batch_metadata)
        print(
            f"Added batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
        )
        time.sleep(sleep_time)  # Sleep to avoid hitting rate limits


# Your existing code to create embeddings and metadata
print("Create a Vector Database from PDF text...")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
texts = [p.page_content for p in pages]  # Assuming 'pages' is previously defined
metadata = pd.DataFrame(index=list(range(len(texts))))
metadata["tag"] = "you-tag"
metadata["title"] = "your-title".encode("utf-8")
vectordb = KDBAI(table, embeddings)  # Make sure 'table' is defined

# Use the rate-limited function to add texts
add_texts_with_rate_limiting(vectordb, texts, metadata)

print("Create LangChain Pipeline...")
qabot = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=TEMP),
    retriever=vectordb.as_retriever(search_kwargs=dict(k=K)),
    return_source_documents=True,
)


Q = "your-query"
# print(f"\n\n{Q}\n")
print(qabot.invoke(dict(query=Q))["result"])
