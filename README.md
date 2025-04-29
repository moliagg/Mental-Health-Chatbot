# Mental-Health-Chatbot
# README: Setting Up Your Environment with Pipenv

## Memory Functionality for LLM Model
This project now includes a memory management system for the LLM model. The memory allows the model to store and retrieve relevant information across interactions.

### How to Use Memory
1. **Initialization**: The memory is initialized in the `connect.py` and `medibot.py` files.
2. **Adding Memory**: You can add new memory items using the `add_memory` method of the `Memory` class.
3. **Retrieving Memory**: Use the `retrieve_memory` method to fetch relevant memory based on user queries.

This functionality enhances the LLM's ability to maintain context and provide more relevant responses.


## Prerequisite: Install Pipenv
Follow the official Pipenv installation guide to set up Pipenv on your system:  
[Install Pipenv Documentation](https://pipenv.pypa.io/en/latest/installation.html)

---

## Steps to Set Up the Environment

### Install Required Packages
Run the following commands in your terminal (assuming Pipenv is already installed):

```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit

