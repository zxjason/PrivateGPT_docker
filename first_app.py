import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

def main():
    
    # Initialize the embeddings and retriever
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    # Initialize the language model and QA system
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Set up Streamlit app
    st.title("Jason's PrivateGPT")
    
    if "answer" not in st.session_state:
        st.session_state.answer = ""
    if "docs" not in st.session_state:
        st.session_state.docs = []
    
    # Get user input
    query = st.text_input("请输入您的问题：")
    
    # Answer the question and display results
    if query:
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], res['source_documents']
        end = time.time()

        st.session_state.answer = answer
        st.session_state.docs = docs

        st.write(f"**问题：** {query}")
        st.write(f"**答案（用时 {round(end - start, 2)} 秒）：** {answer}")

        if docs:
            st.write("**相关来源：**")
            for document in docs:
                st.write(f"{document.metadata['source']}:")
                st.write(document.page_content)

if __name__ == "__main__":
    main()