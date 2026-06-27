import os

import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question

# set the working directory

working_dir = os.path.dirname(os.path.abspath(__file__))
print("working_dire:",working_dir)

# set the streamlit app title
st.title("Document RAG Application")

# create file uploader widget so that user can upload the file

uploaded_file = st.file_uploader("Upload a PDF file",type=['pdf'])

if uploaded_file is not None:
    # define save path
    save_path = os.path.join(working_dir,uploaded_file.name)

    # save the file
    with open(save_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chroma_db(uploaded_file.name)
    st.info("Document Processed successfully")

user_question = st.text_area("Ask your question about document")

if st.button("Answer"):
    answer = answer_question(user_question)
    st.markdown("Answer from uploaded Document")
    st.markdown(answer)