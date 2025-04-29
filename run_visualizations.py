import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Set style for better visualizations
if 'seaborn' in plt.style.available:
    plt.style.use('seaborn')
else:
    plt.style.use('default')  # Fallback to default style
sns.set_palette("husl")

# Read the datasets
print("Loading datasets...")
faq_df = pd.read_csv('data/Updated_Mental_Health_FAQ.csv')
chatbot_data = pd.read_csv('data/mental_health_chatbot_data.csv')
movies_df = pd.read_csv('data/mental_health_movies_large.csv')
music_df = pd.read_csv('data/therapy_music_recommendations.csv')
professionals_df = pd.read_csv('data/professionals.csv')

# Print dataset info
print(f"FAQ dataset shape: {faq_df.shape}")
print(f"Chatbot dataset shape: {chatbot_data.shape}")
print(f"Movies dataset shape: {movies_df.shape}")
print(f"Music dataset shape: {music_df.shape}")
print(f"Professionals dataset shape: {professionals_df.shape}")

# Create output directory for saving figures
os.makedirs('visualization_output', exist_ok=True)

def run_visualization(segment_number):
    """Run a specific visualization segment"""
    print(f"\n{'='*50}")
    print(f"Running Segment {segment_number}")
    print(f"{'='*50}")
    
    if segment_number == 1:
        # 1. FAQ Analysis
        print("Generating FAQ Question Distribution...")
        plt.figure(figsize=(12, 6))
        # Since there's no Category column, let's analyze question length distribution
        faq_df['question_length'] = faq_df['Questions'].str.len()
        sns.histplot(faq_df['question_length'], bins=20, kde=True)
        plt.title('Distribution of FAQ Question Lengths')
        plt.xlabel('Question Length (characters)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('visualization_output/faq_question_length.png')
        plt.close()
        print("✓ FAQ Question Length visualization completed")
        
    elif segment_number == 2:
        # 2. Chatbot Data Analysis
        print("Generating Chatbot Query Analysis...")
        plt.figure(figsize=(12, 6))
        # Since there's no intent column, let's analyze user query length
        chatbot_data['query_length'] = chatbot_data['User Query'].str.len()
        sns.histplot(chatbot_data['query_length'], bins=20, kde=True)
        plt.title('Distribution of Chatbot User Query Lengths')
        plt.xlabel('Query Length (characters)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('visualization_output/chatbot_query_length.png')
        plt.close()
        print("✓ Chatbot Query Length visualization completed")
        
    elif segment_number == 3:
        # 3. Movie Analysis
        print("Generating Top Rated Mental Health Movies...")
        plt.figure(figsize=(15, 7))
        # Top 10 highest rated mental health movies using user_rating column
        top_movies = movies_df.nlargest(10, 'user_rating')
        sns.barplot(x='user_rating', y='title', data=top_movies)
        plt.title('Top 10 Highest Rated Mental Health Movies')
        plt.xlabel('Rating')
        plt.ylabel('Movie Title')
        plt.tight_layout()
        plt.savefig('visualization_output/top_movies.png')
        plt.close()
        print("✓ Top Movies visualization completed")
        
        # Additional movie visualization - genre distribution
        plt.figure(figsize=(12, 8))
        # Extract genres (assuming they might be in a list-like format)
        all_genres = []
        for genre_list in movies_df['genre'].str.split(','):
            if isinstance(genre_list, list):
                all_genres.extend([g.strip() for g in genre_list])
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        sns.barplot(x=genre_counts.values, y=genre_counts.index)
        plt.title('Top 10 Movie Genres in Mental Health Movies')
        plt.xlabel('Count')
        plt.ylabel('Genre')
        plt.tight_layout()
        plt.savefig('visualization_output/movie_genres.png')
        plt.close()
        print("✓ Movie Genres visualization completed")
        
    elif segment_number == 4:
        # 4. Music Recommendations Analysis
        print("Generating Music Analysis...")
        # Since there's no Genre column, let's analyze mental health tags
        plt.figure(figsize=(12, 8))
        all_tags = []
        for tag_list in music_df['mental_health_tags'].str.split(','):
            if isinstance(tag_list, list):
                all_tags.extend([t.strip() for t in tag_list])
        
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        sns.barplot(x=tag_counts.values, y=tag_counts.index)
        plt.title('Top 10 Mental Health Tags in Therapeutic Music')
        plt.xlabel('Count')
        plt.ylabel('Mental Health Tag')
        plt.tight_layout()
        plt.savefig('visualization_output/music_mental_health_tags.png')
        plt.close()
        print("✓ Music Mental Health Tags visualization completed")
        
        # Additional music visualization - mood score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(music_df['mood_score'], bins=20, kde=True)
        plt.title('Distribution of Mood Scores in Therapeutic Music')
        plt.xlabel('Mood Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('visualization_output/music_mood_scores.png')
        plt.close()
        print("✓ Music Mood Scores visualization completed")
        
    elif segment_number == 5:
        # 5. Professionals Analysis
        print("Generating Mental Health Professional Specializations...")
        plt.figure(figsize=(10, 6))
        # Using lowercase column name 'specialization' instead of 'Specialization'
        specializations = professionals_df['specialization'].value_counts().head(10)
        plt.pie(specializations.values, labels=specializations.index, autopct='%1.1f%%')
        plt.title('Distribution of Mental Health Professional Specializations')
        plt.axis('equal')
        plt.savefig('visualization_output/professional_specializations.png')
        plt.close()
        print("✓ Professional Specializations visualization completed")
        
        # Additional professionals visualization - profession distribution
        plt.figure(figsize=(12, 6))
        professions = professionals_df['profession'].value_counts()
        sns.barplot(x=professions.values, y=professions.index)
        plt.title('Distribution of Mental Health Professions')
        plt.xlabel('Count')
        plt.ylabel('Profession')
        plt.tight_layout()
        plt.savefig('visualization_output/professional_professions.png')
        plt.close()
        print("✓ Professional Professions visualization completed")
        
    elif segment_number == 6:
        # 6. Interactive Visualization using Plotly
        print("Generating Interactive Visualizations...")
        
        # Movie sentiment analysis
        fig = px.scatter(movies_df, x='user_rating', y='sentiment_score', 
                         hover_name='title', color='release_year',
                         title='Movie Ratings vs. Sentiment Scores')
        fig.write_html('visualization_output/movie_sentiment_analysis.html')
        print("✓ Movie Sentiment Analysis visualization completed (saved as HTML)")
        
        # Music popularity vs energy
        fig = px.scatter(music_df, x='popularity', y='energy', 
                         hover_name='title', color='mood_score',
                         title='Music Popularity vs. Energy Levels')
        fig.write_html('visualization_output/music_popularity_energy.html')
        print("✓ Music Popularity vs Energy visualization completed (saved as HTML)")
        
    elif segment_number == 7:
        # 7. Correlation Analysis for Movies
        print("Generating Movie Correlation Analysis...")
        # Select only numeric columns
        numeric_cols = movies_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            correlation_matrix = movies_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix of Movie Numeric Features')
            plt.tight_layout()
            plt.savefig('visualization_output/movie_correlation.png')
            plt.close()
            print("✓ Movie Correlation Analysis visualization completed")
        else:
            print("✗ Not enough numeric columns for correlation analysis")
            
    elif segment_number == 8:
        # 8. Correlation Analysis for Music
        print("Generating Music Correlation Analysis...")
        # Select only numeric columns
        numeric_cols = music_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            correlation_matrix = music_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix of Music Numeric Features')
            plt.tight_layout()
            plt.savefig('visualization_output/music_correlation.png')
            plt.close()
            print("✓ Music Correlation Analysis visualization completed")
        else:
            print("✗ Not enough numeric columns for correlation analysis")

# Run all visualization segments
if __name__ == "__main__":
    print("Starting visualization process...")
    for segment in range(1, 9):
        try:
            run_visualization(segment)
        except Exception as e:
            print(f"Error in segment {segment}: {str(e)}")
    print("\nAll visualizations completed. Check the 'visualization_output' directory for results.")
