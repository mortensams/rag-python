import openai
import textwrap
import sys

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

openai.api_key = "sk-proj-6WuYDjQJEuDYs0ZMHcE6T3BlbkFJCynlMic8wR9kW9TWxnel"

# get question
question = sys.argv[1] 
print("Your question was: " + question)

# load documents using Llamaindex
documents = SimpleDirectoryReader("./documents/").load_data()
print("Document ID:", documents[0].doc_id)

# Create an index over the documnts
vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1536, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

query_engine = index.as_query_engine()
response = query_engine.query(question)
print(textwrap.fill(str(response), 100))