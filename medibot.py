import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Import the Memory class from connect.py (which itself imports from create_memory_for_llm.py)
from connect import interact_with_llm, memory

# Load environment variables - make sure this runs early
load_dotenv(verbose=True)

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model,
                          allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=[
                            "context", "question"])
    return prompt


def load_llm(huggingface_repo_id, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token,
                      "max_length": "512"}
    )
    return llm


def main():
    st.title("Mental Health Chatbot")
    st.subheader("Ask me anything about mental health!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your message here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question about mental health.
                Use the memory provided to maintain context of the conversation.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context.

                Context: {context}
                Question: {question}
                """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        # Get token from environment variable properly
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        # If token not found in environment variables, use the direct value from the .env file
        if not HF_TOKEN:
            HF_TOKEN = "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"
            st.info("Using fallback HuggingFace token")
        
        # For testing without HuggingFace
        use_mock = False  # Set to True to use a mock response instead of actual API call

        try:
            # First check if there's relevant memory
            memory_response = memory.retrieve_memory(prompt)
            
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(
                    huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={
                    'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Pass the memory context along with the query
            response = qa_chain.invoke({
                'query': prompt,
                'context': memory_response if memory_response else ""
            })

            result = response["result"]
            
            # Add this interaction to memory
            memory.add_memory(f"User: {prompt}\nAssistant: {result}")
            
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append(
                {'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
