import pandas as pd
import logging
from langchain_community.document_loaders import DirectoryLoader
import re
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS

# Memory class to manage LLM memory
class Memory:
    def __init__(self, max_memory_items=50):
        self.memory_store = []
        self.max_memory_items = max_memory_items  # Limit memory size to prevent excessive growth

    def add_memory(self, memory_item):
        """
        Add a new memory item to the store
        
        Args:
            memory_item (str): The memory to add
        """
        # Add timestamp to memory for better tracking
        # Add the memory to the beginning of the list (most recent first)
        self.memory_store.insert(0, memory_item)
        
        # Prevent memory from growing too large
        if len(self.memory_store) > self.max_memory_items:
            self.memory_store = self.memory_store[:self.max_memory_items]
            
        logger.info(f"Added memory: {memory_item[:50]}...")

    def retrieve_memory(self, query, max_results=5):
        """
        Retrieve relevant memories based on a query
        
        Args:
            query (str): The query to search for
            max_results (int): Maximum number of results to return
            
        Returns:
            str: Concatenated relevant memories
        """
        if not self.memory_store:
            return ""
            
        # For simple implementation, use keyword matching
        query_keywords = self._extract_keywords(query.lower())
        
        # Score each memory based on keyword matches
        scored_memories = []
        for memory in self.memory_store:
            score = 0
            memory_lower = memory.lower()
            
            # Score based on keyword matches
            for keyword in query_keywords:
                if keyword in memory_lower:
                    score += 1
                    
            # Prioritize recent memories (earlier in the list)
            position_boost = 1.0 / (self.memory_store.index(memory) + 1)
            score += position_boost
                    
            if score > 0:
                scored_memories.append((memory, score))
                
        # Sort by score in descending order
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Take top results
        top_memories = [memory for memory, _ in scored_memories[:max_results]]
        
        # If no matches found but we have memories, return the most recent one for context
        if not top_memories and self.memory_store:
            top_memories = [self.memory_store[0]]
            
        return "\n".join(top_memories)
    
    def _extract_keywords(self, text):
        """Extract keywords from text"""
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def clear_memory(self):
        """Clear all memories"""
        self.memory_store = []
        logger.info("Memory cleared")

# Step 1: Load CSV files
DATA_PATH="data/"
def load_csv_files(data):
    # Load all CSV files in the directory
    dfs = []
    for csv_file in ['mental_health_chatbot_data.csv', 'Mental_Health_FAQ.csv', 'train.csv', 'professionals.csv', 'mental_health_movies_large.csv' , 'therapy_music_recommendations.csv']:
        df = pd.read_csv(f"{data}/{csv_file}")
        dfs.append(df)
    
    # Combine all dataframes and convert to list of strings
    combined_df = pd.concat(dfs)
    documents = combined_df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()
    return documents

documents = load_csv_files(data=DATA_PATH)
logger.info(f"Loaded {len(documents)} documents from CSV files")



# Step 2: Create Chunks
def create_chunks(text_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    # Convert list of strings to documents format
    from langchain_core.documents import Document
    documents = [Document(page_content=text) for text in text_data]
    text_chunks=text_splitter.split_documents(documents)
    return text_chunks


text_chunks=create_chunks(text_data=documents)
logger.info(f"Created {len(text_chunks)} text chunks")



# Step 3: Create Vector Embeddings 
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
logger.info(f"Saved FAISS vector store with {len(text_chunks)} embeddings to {DB_FAISS_PATH}")


# Initialize memory
memory = Memory()

# Add some initial memories about the system
memory.add_memory("This is a mental health chatbot designed to provide information and support for mental health related queries.")
memory.add_memory("The chatbot uses data from mental health resources, FAQs, professional advice, and therapeutic content.")
memory.add_memory("The chatbot should be empathetic, supportive, and provide factual information about mental health topics.")
