import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from create_memory_for_llm import Memory  # Import the Memory class
from langchain_community.vectorstores import FAISS

# Define constants
DB_FAISS_PATH = "vectorstore/db_faiss"

# Initialize memory
memory = Memory()

def get_vectorstore():
    """Load the FAISS vectorstore for retrieval"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model,
                            allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def load_llm(huggingface_repo_id, hf_token):
    """Initialize the HuggingFace LLM endpoint"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            model_kwargs={"token": hf_token,
                        "max_length": "512"}
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def interact_with_llm(user_input):
    """
    Process user input with LLM, incorporating memory
    
    Args:
        user_input (str): The user's query or message
        
    Returns:
        str: The LLM's response
    """
    try:
        # Get HuggingFace token
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            hf_token = "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"
            print("Using fallback HuggingFace token")
        
        # Retrieve relevant memory based on user input
        relevant_memory = memory.retrieve_memory(user_input)
        
        # Get vectorstore
        vectorstore = get_vectorstore()
        if not vectorstore:
            return "Error: Could not load the vectorstore."
        
        # Set up prompt template with memory context
        custom_prompt_template = """
        Use the pieces of information provided in the context to answer user's question about mental health.
        Consider the conversation history and memory provided to maintain context.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        Memory/Prior Conversation: {memory}
        Context: {context}
        Question: {question}
        """
        
        prompt = PromptTemplate(
            template=custom_prompt_template, 
            input_variables=["memory", "context", "question"]
        )
        
        # Set up retrieval chain
        huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(huggingface_repo_id=huggingface_repo_id, hf_token=hf_token),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        
        # Invoke the chain with the user input and memory
        response = qa_chain.invoke({
            'query': user_input,
            'memory': relevant_memory if relevant_memory else "",
        })
        
        # Store the interaction in memory
        memory.add_memory(f"User: {user_input}\nAssistant: {response['result']}")
        
        return response["result"]
    
    except Exception as e:
        print(f"An error occurred in interact_with_llm: {e}")
        return f"Sorry, I couldn't process your request. Error: {str(e)}"
