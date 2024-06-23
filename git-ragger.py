import openai
import textwrap
import sys
import os
from pymilvus import MilvusClient

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

# set the openai api key
openai.api_key = os.getenv("OPENAPI_API_KEY")
milvus_host = os.getenv("MILVUS_HOST")
document_directory = os.getenv("DOCUMENT_DIRECTORY")

if not milvus_host: milvus_host = "./milvus_demo.db"
if not document_directory: document_directory="./documents"
print("Milvus host is:" + milvus_host)
# load documents using Llamaindex
documents = SimpleDirectoryReader(document_directory).load_data()

client = MilvusClient("./milvus_demo.db")

# Create an index over the documnts
vector_store = MilvusVectorStore(uri=milvus_host, dim=1536,collection_name="git_ragger")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print("Done indexing " + documents.count() + "files")
