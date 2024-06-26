from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
import openai
import gradio as gr
import os
from dotenv import load_dotenv
import random
import time

# set the openai api key
openai.api_key = os.getenv("OPENAPI_API_KEY")
milvus_host = os.getenv("MILVUS_HOST")

if not milvus_host: milvus_host = "./milvus_demo.db"

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        vector_store = MilvusVectorStore(uri=milvus_host, collection_name="git_ragger")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        query_engine = index.as_query_engine()
        response = query_engine.query(message)
        chat_history.append((message, response.response))
        return "",chat_history
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)