import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import sys

# Set style for visualizations
plt.style.use('fivethirtyeight')
sns.set_palette("husl")

# Constants
OUTPUT_DIR = "report_assets"
REPORT_FILENAME = "Mental_Health_Chatbot_Project_Report.pdf"

# Create directory for report assets if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PDF Class with additional features
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.width = 210
        self.height = 297
        self.set_auto_page_break(auto=True, margin=15)
        # Using standard fonts instead of DejaVu to avoid font dependency issues
        
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Mental Health Chatbot Project Report', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        # Sanitize text for PDF
        title = self._sanitize_text(title)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        # Sanitize text for PDF
        body = self._sanitize_text(body)
        self.multi_cell(0, 10, body)
        self.ln()
        
    def add_image(self, img_path, w=None, h=None, caption=None):
        if w is None:
            w = self.width - 40
        
        if os.path.exists(img_path):
            self.image(img_path, x=10, w=w)
            
            if caption:
                self.set_font('Arial', 'B', 10)
                # Sanitize text for PDF
                caption = self._sanitize_text(caption)
                self.cell(0, 10, caption, 0, 1, 'C')
        else:
            self.set_font('Arial', '', 10)
            self.cell(0, 10, f"Image not found: {img_path}", 0, 1, 'C')
            
        self.ln(5)
        
    def section_heading(self, heading):
        self.set_font('Arial', 'B', 14)
        # Sanitize text for PDF
        heading = self._sanitize_text(heading)
        self.cell(0, 10, heading, 0, 1, 'L')
        self.ln(2)
        
    def subsection_heading(self, heading):
        self.set_font('Arial', 'B', 12)
        # Sanitize text for PDF
        heading = self._sanitize_text(heading)
        self.cell(0, 10, heading, 0, 1, 'L')
        self.ln(2)
        
    def add_table(self, headers, data, col_widths=None):
        # Default column widths if none provided
        if col_widths is None:
            col_widths = [self.width / len(headers) - 10] * len(headers)
        
        # Headers
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        
        for i, header in enumerate(headers):
            # Sanitize text for PDF
            header = self._sanitize_text(header)
            self.cell(col_widths[i], 10, header, 1, 0, 'C', 1)
        self.ln()
        
        # Data
        self.set_font('Arial', '', 10)
        self.set_fill_color(255, 255, 255)
        
        alt_color = False
        for row in data:
            if alt_color:
                self.set_fill_color(240, 240, 240)
            else:
                self.set_fill_color(255, 255, 255)
                
            for i, cell in enumerate(row):
                # Sanitize text for PDF
                cell_text = self._sanitize_text(str(cell))
                self.cell(col_widths[i], 10, cell_text, 1, 0, 'L', 1)
            self.ln()
            alt_color = not alt_color
            
    def _sanitize_text(self, text):
        """Sanitize text to handle encoding issues"""
        # Replace problematic characters
        text = text.replace('•', '-')  # Replace bullets with hyphens
        
        # Handle other special characters
        result = ''
        for char in text:
            if ord(char) < 128:
                result += char
            else:
                result += ' '  # Replace non-ASCII chars with space
                
        return result

# Generate Charts and Save Them
def generate_visualizations():
    print("Generating visualizations for the report...")
    
    # Load datasets
    try:
        faq_df = pd.read_csv('data/Mental_Health_FAQ.csv')
        chatbot_data = pd.read_csv('data/mental_health_chatbot_data.csv')
        movies_df = pd.read_csv('data/mental_health_movies_large.csv')
        music_df = pd.read_csv('data/therapy_music_recommendations.csv')
        professionals_df = pd.read_csv('data/professionals.csv')
        
        # 1. FAQ Question Length Distribution
        plt.figure(figsize=(10, 6))
        faq_df['question_length'] = faq_df['Questions'].str.len()
        sns.histplot(faq_df['question_length'], bins=20, kde=True, color='pink')
        plt.title('Distribution of FAQ Question Lengths')
        plt.xlabel('Question Length (characters)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/faq_question_length.png', dpi=300)
        plt.close()
        
        # 2. Chatbot Query Analysis
        plt.figure(figsize=(10, 6))
        chatbot_data['query_length'] = chatbot_data['User Query'].str.len()
        sns.histplot(chatbot_data['query_length'], bins=20, kde=True, color='pink')
        plt.title('Distribution of Chatbot User Query Lengths')
        plt.xlabel('Query Length (characters)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/chatbot_query_length.png', dpi=300)
        plt.close()
        
        # 3. Movie Analysis - Top Rated
        plt.figure(figsize=(12, 8))
        top_movies = movies_df.nlargest(10, 'user_rating')
        ax = sns.barplot(x='user_rating', y='title', data=top_movies, palette='husl')
        ax.bar_label(ax.containers[0], fmt='%.1f')
        plt.title('Top 10 Highest Rated Mental Health Movies')
        plt.xlabel('Rating')
        plt.ylabel('Movie Title')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/top_movies.png', dpi=300)
        plt.close()
        
        # 4. Movie Genre Distribution
        plt.figure(figsize=(12, 8))
        all_genres = []
        for genre_list in movies_df['genre'].str.split(','):
            if isinstance(genre_list, list):
                all_genres.extend([g.strip() for g in genre_list])
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        ax = sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='husl')
        ax.bar_label(ax.containers[0])
        plt.title('Top 10 Movie Genres in Mental Health Movies')
        plt.xlabel('Count')
        plt.ylabel('Genre')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/movie_genres.png', dpi=300)
        plt.close()
        
        # 5. Music Analysis - Mental Health Tags
        plt.figure(figsize=(12, 8))
        all_tags = []
        for tag_list in music_df['mental_health_tags'].str.split(','):
            if isinstance(tag_list, list):
                all_tags.extend([t.strip() for t in tag_list])
        
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        ax = sns.barplot(x=tag_counts.values, y=tag_counts.index, palette='husl')
        ax.bar_label(ax.containers[0])
        plt.title('Top 10 Mental Health Tags in Therapeutic Music')
        plt.xlabel('Count')
        plt.ylabel('Mental Health Tag')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/music_mental_health_tags.png', dpi=300)
        plt.close()
        
        # 6. Professionals Analysis - Specializations
        plt.figure(figsize=(12, 8))
        specializations = professionals_df['specialization'].value_counts().head(10)
        plt.pie(specializations.values, labels=specializations.index, autopct='%1.1f%%', 
                shadow=True, startangle=90, colors=sns.color_palette("husl", len(specializations)))
        plt.axis('equal')
        plt.title('Distribution of Mental Health Professional Specializations')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/professional_specializations.png', dpi=300)
        plt.close()
        
        # 7. Correlation Matrix - Movies
        plt.figure(figsize=(10, 8))
        numeric_cols = movies_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            correlation_matrix = movies_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix of Movie Numeric Features')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/movie_correlation.png', dpi=300)
            plt.close()
        
        # 8. Correlation Matrix - Music
        plt.figure(figsize=(10, 8))
        numeric_cols = music_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            correlation_matrix = music_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix of Music Numeric Features')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/music_correlation.png', dpi=300)
            plt.close()
            
        # 9. Professional Types Distribution
        plt.figure(figsize=(12, 8))
        professions = professionals_df['profession'].value_counts()
        ax = sns.barplot(x=professions.values, y=professions.index, palette='husl')
        ax.bar_label(ax.containers[0])
        plt.title('Distribution of Mental Health Professions')
        plt.xlabel('Count')
        plt.ylabel('Profession')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/professional_professions.png', dpi=300)
        plt.close()
        
        # 10. Generate a sample plotly figure and save as PNG
        try:
            fig = px.scatter(movies_df, x='user_rating', y='sentiment_score', 
                         hover_name='title', color='release_year',
                         title='Movie Ratings vs. Sentiment Scores')
            pio.write_image(fig, f'{OUTPUT_DIR}/movie_sentiment_interactive.png', scale=2)
        except Exception as e:
            print(f"Skipping Plotly visualization: {e}")
            
        print("All visualizations generated successfully!")
        return True
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return False

# Create the PDF Report
def create_pdf_report():
    print("Creating PDF report...")
    pdf = PDF()
    
    # Cover Page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.ln(60)
    pdf.cell(0, 10, 'Mental Health Chatbot', 0, 1, 'C')
    pdf.ln(10)
    pdf.cell(0, 10, 'Project Report', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
    
    # Table of Contents
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    
    contents = [
        "1. Executive Summary",
        "2. Project Overview",
        "3. Data Analysis",
        "4. Chatbot Architecture",
        "5. Mental Health Resources",
        "6. Technical Implementation",
        "7. Conclusion and Future Work"
    ]
    
    for item in contents:
        pdf.chapter_body(item)
        pdf.ln(5)
        
    # 1. Executive Summary
    pdf.add_page()
    pdf.chapter_title("1. Executive Summary")
    summary_text = """
This report presents a comprehensive overview of the Mental Health Chatbot project, an AI-powered 
solution designed to provide mental health support and resources. The chatbot utilizes natural 
language processing to understand user queries, provide appropriate responses, and offer 
recommendations for mental health resources including professionals, movies, and music.

Key components of the project include:
• A trained language model for natural conversation
• Integration with mental health FAQs and resources
• Data-driven recommendations for therapeutic content
• User-friendly interface via Streamlit
• Secure memory system for conversation context

The following pages detail the project architecture, data analysis, technical implementation, 
and potential areas for future enhancement.
"""
    pdf.chapter_body(summary_text)
    
    # 2. Project Overview
    pdf.add_page()
    pdf.chapter_title("2. Project Overview")
    overview_text = """
The Mental Health Chatbot project aims to create an accessible, AI-powered solution for providing 
mental health support and resources. Mental health issues affect millions globally, yet access to 
immediate support remains limited. This chatbot serves as a bridge, offering instant responses 
to mental health queries, providing resources, and directing users to professional help when needed.

Project Objectives:
• Create a responsive chatbot for mental health support
• Provide evidence-based answers to mental health questions
• Offer personalized recommendations for therapeutic content
• Connect users with appropriate mental health professionals
• Ensure user privacy and data security

The system integrates various components including a language model, vector database for efficient 
information retrieval, and curated datasets of mental health resources. The chatbot is accessible 
through a Streamlit web interface, making it easy for users to interact with the system.
"""
    pdf.chapter_body(overview_text)
    
    # 3. Data Analysis
    pdf.add_page()
    pdf.chapter_title("3. Data Analysis")
    data_analysis_text = """
The Mental Health Chatbot project relies on several datasets to provide comprehensive mental 
health support. This section presents analysis of these datasets, highlighting key patterns 
and insights that inform the chatbot's functionality.
"""
    pdf.chapter_body(data_analysis_text)
    
    # FAQ Analysis
    pdf.section_heading("3.1 FAQ Analysis")
    faq_text = """
The Mental Health FAQ dataset forms the foundation of the chatbot's knowledge base. It contains 
a wide range of questions and answers related to mental health topics. The following visualization 
shows the distribution of question lengths in the FAQ dataset.
"""
    pdf.chapter_body(faq_text)
    pdf.add_image(f'{OUTPUT_DIR}/faq_question_length.png', 
                  caption="Figure 1: Distribution of FAQ Question Lengths")
    
    # Chatbot Query Analysis
    pdf.section_heading("3.2 Chatbot Query Analysis")
    query_text = """
Understanding user query patterns is essential for optimizing the chatbot's performance. The 
visualization below shows the distribution of query lengths from user interactions. This information 
helps in training and fine-tuning the model to better handle typical user inputs.
"""
    pdf.chapter_body(query_text)
    pdf.add_image(f'{OUTPUT_DIR}/chatbot_query_length.png', 
                  caption="Figure 2: Distribution of Chatbot User Query Lengths")
    
    # 4. Chatbot Architecture
    pdf.add_page()
    pdf.chapter_title("4. Chatbot Architecture")
    architecture_text = """
The Mental Health Chatbot employs a sophisticated architecture that integrates several components 
to deliver accurate, helpful responses to user queries. The system is built using modern NLP 
techniques and follows best practices in conversational AI design.

Key components of the architecture include:

1. User Interface (Streamlit):
   The frontend is built with Streamlit, providing an intuitive interface for users to interact 
   with the chatbot. The interface maintains conversation history, displays responses clearly, 
   and offers additional resource recommendations.

2. Language Model:
   The core of the chatbot is powered by a Hugging Face language model, accessed through the 
   LangChain framework. This model processes user inputs, understands intent, and generates 
   natural language responses.

3. Vector Database (FAISS):
   For efficient information retrieval, the system uses FAISS (Facebook AI Similarity Search) 
   to store and query vector embeddings of the knowledge base. This allows the chatbot to 
   quickly find relevant information when responding to user queries.

4. Memory System:
   A dedicated memory component stores conversation context, enabling the chatbot to maintain 
   coherent, contextually appropriate conversations over time. The memory system uses vector 
   embeddings to retrieve relevant past conversations.

5. Resource Databases:
   Curated databases of mental health professionals, therapeutic movies, and music recommendations 
   enable the chatbot to provide personalized resource suggestions based on user needs.

The following diagram illustrates the flow of information through the system:

1. User inputs a query through the Streamlit interface
2. The query is processed and embedded using the Hugging Face embedding model
3. Relevant information is retrieved from the FAISS vector store
4. The context and query are passed to the language model for response generation
5. The response is displayed to the user, along with any relevant resource recommendations
6. Conversation history is updated in the memory system
"""
    pdf.chapter_body(architecture_text)
    
    # 5. Mental Health Resources
    pdf.add_page()
    pdf.chapter_title("5. Mental Health Resources")
    
    # Movies Analysis
    pdf.section_heading("5.1 Therapeutic Movies")
    movies_text = """
The project includes a curated database of movies that address mental health themes. These films 
can provide therapeutic value, increase understanding, and reduce stigma around mental health issues. 
The following visualizations highlight key insights from the movie database.
"""
    pdf.chapter_body(movies_text)
    
    pdf.add_image(f'{OUTPUT_DIR}/top_movies.png', 
                  caption="Figure 3: Top 10 Highest Rated Mental Health Movies")
                  
    pdf.add_image(f'{OUTPUT_DIR}/movie_genres.png',
                  caption="Figure 4: Top 10 Movie Genres in Mental Health Movies")
                  
    pdf.add_image(f'{OUTPUT_DIR}/movie_correlation.png',
                  caption="Figure 5: Correlation Matrix of Movie Numeric Features")
    
    # Music Analysis
    pdf.add_page()
    pdf.section_heading("5.2 Therapeutic Music")
    music_text = """
Music therapy is increasingly recognized as an effective intervention for various mental health 
conditions. The project includes a dataset of music recommendations tailored to different mental 
health needs. The following visualizations showcase patterns in the therapeutic music database.
"""
    pdf.chapter_body(music_text)
    
    pdf.add_image(f'{OUTPUT_DIR}/music_mental_health_tags.png',
                  caption="Figure 6: Top Mental Health Tags in Therapeutic Music")
                  
    pdf.add_image(f'{OUTPUT_DIR}/music_correlation.png',
                  caption="Figure 7: Correlation Matrix of Music Numeric Features")
    
    # Professionals Analysis
    pdf.add_page()
    pdf.section_heading("5.3 Mental Health Professionals")
    professionals_text = """
Access to appropriate professional help is crucial for mental health support. The project 
includes a database of mental health professionals with various specializations. The visualizations 
below illustrate the distribution of professionals in the database.
"""
    pdf.chapter_body(professionals_text)
    
    pdf.add_image(f'{OUTPUT_DIR}/professional_specializations.png',
                  caption="Figure 8: Distribution of Mental Health Professional Specializations")
                  
    pdf.add_image(f'{OUTPUT_DIR}/professional_professions.png',
                  caption="Figure 9: Distribution of Mental Health Professions")
    
    # 6. Technical Implementation
    pdf.add_page()
    pdf.chapter_title("6. Technical Implementation")
    technical_text = """
The Mental Health Chatbot is implemented using a combination of cutting-edge technologies 
and frameworks. This section details the technical aspects of the implementation.

Key Technologies:
• Python 3.9+: Primary programming language
• Streamlit: Web interface for chatbot interaction
• LangChain: Framework for working with language models
• Hugging Face Transformers: Source of the language and embedding models
• FAISS: Vector store for efficient semantic search
• Pandas: Data manipulation and analysis
• Matplotlib/Seaborn/Plotly: Data visualization
• PyPDF2: PDF generation for reports

The implementation follows a modular approach, with separate components for:
1. Data processing and vectorization
2. Language model interaction
3. Memory management
4. Web interface
5. Resource recommendation

The system architecture prioritizes:
• Scalability: The system can handle increasing numbers of users
• Modularity: Components can be upgraded individually
• Security: User data is handled securely
• Responsiveness: Quick response times for user queries
"""
    pdf.chapter_body(technical_text)
    
    pdf.section_heading("6.1 Code Structure")
    code_structure_text = """
The project codebase is organized as follows:

• medibot.py: Main Streamlit application for the chatbot interface
• connect.py: Handles connections between components and the LLM
• create_memory_for_llm.py: Implements the memory system for context retention
• run_visualizations.py: Generates visualizations for data analysis
• /data/: Directory containing all datasets
  - mental_health_chatbot_data.csv: User interaction data
  - Mental_Health_FAQ.csv: Frequently asked questions
  - mental_health_movies_large.csv: Therapeutic movie database
  - therapy_music_recommendations.csv: Therapeutic music database
  - professionals.csv: Mental health professionals database
• /vectorstore/: Directory for FAISS vector indices
"""
    pdf.chapter_body(code_structure_text)
    
    # 7. Conclusion and Future Work
    pdf.add_page()
    pdf.chapter_title("7. Conclusion and Future Work")
    conclusion_text = """
The Mental Health Chatbot project demonstrates the potential of AI to provide accessible mental 
health support and resources. By combining natural language processing with curated mental health 
information, the system offers a valuable tool for users seeking information, support, and resources.

Key Achievements:
• Development of a responsive, empathetic chatbot for mental health support
• Integration of comprehensive mental health resources
• Data-driven insights into mental health content and user interactions
• Secure, context-aware conversation system

Future Work:
Several avenues for future enhancement have been identified:

1. Model Improvements:
   • Fine-tuning the language model on specialized mental health corpora
   • Implementing more sophisticated emotion detection
   • Adding multilingual support

2. Feature Enhancements:
   • Integration with telehealth services for professional consultations
   • Development of personalized mental health plans
   • Addition of interactive cognitive behavioral therapy exercises
   • Implementing voice interaction capabilities

3. Data Expansion:
   • Expanding the database of therapeutic content
   • Adding more region-specific professional resources
   • Incorporating user feedback for continuous improvement

4. Technical Upgrades:
   • Deploying to cloud infrastructure for wider accessibility
   • Implementing enhanced security measures
   • Developing mobile applications for iOS and Android

The Mental Health Chatbot represents a starting point for AI-assisted mental health support. 
With continued development and refinement, such systems could play an increasingly important 
role in addressing the global mental health crisis by providing accessible, immediate support 
to those in need.
"""
    pdf.chapter_body(conclusion_text)
    
    # Save the PDF
    pdf.output(REPORT_FILENAME)
    print(f"PDF report generated successfully: {REPORT_FILENAME}")
    return True

# Main function
def main():
    print("Starting report generation process...")
    
    # Step 1: Generate visualizations
    visualizations_success = generate_visualizations()
    
    # Step 2: Create PDF report
    if visualizations_success:
        create_pdf_report()
        print(f"Report generated successfully. Output file: {REPORT_FILENAME}")
    else:
        print("Failed to generate visualizations. Report creation aborted.")

if __name__ == "__main__":
    main()
