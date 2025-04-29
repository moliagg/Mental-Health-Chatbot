import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure output directory exists
os.makedirs('visualization_output', exist_ok=True)

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 10))

# Set background to light gray
ax.set_facecolor('#f5f5f5')

# Function to create a box with title
def create_box(x, y, width, height, title, color, alpha=0.7):
    box = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black',
                           facecolor=color, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + width/2, y + height - 0.3, title, ha='center', va='center', 
            fontsize=12, fontweight='bold')
    return box

# Function to create component within a box
def add_component(x, y, name, parent_width):
    comp = patches.Rectangle((x, y), parent_width - 0.6, 0.5, linewidth=1,
                           edgecolor='black', facecolor='white', alpha=0.9)
    ax.add_patch(comp)
    ax.text(x + (parent_width - 0.6)/2, y + 0.25, name, ha='center', va='center', fontsize=10)
    return comp

# Function to draw an arrow
def draw_arrow(start_x, start_y, end_x, end_y, color='black', style='->', width=1):
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle=style, color=color, lw=width))

# Main systems
data_sources = create_box(1, 8, 4, 2.5, 'Data Sources', '#ffcccc')
preprocessing = create_box(6, 8, 4, 2.5, 'Data Processing Pipeline', '#ccffcc')
vector_store = create_box(11, 8, 4, 2.5, 'Vector Store & Embeddings', '#ccccff')

llm_system = create_box(3.5, 4.5, 4, 2, 'LLM System', '#ffffcc')
memory_system = create_box(8.5, 4.5, 4, 2, 'Memory System', '#ffccff')

user_interface = create_box(6, 1, 4, 2, 'User Interface', '#ccffff')

# Data source components
add_component(1.2, 9.7, 'Mental Health FAQ Dataset', 4)
add_component(1.2, 9.1, 'Chatbot Training Data', 4)
add_component(1.2, 8.5, 'Mental Health Movies Dataset', 4)
add_component(1.2, 7.9, 'Therapeutic Music Dataset', 4)
add_component(1.2, 7.3, 'Professional Information Dataset', 4)

# Data processing components
add_component(6.2, 9.7, 'Data Loading (load_csv_files)', 4)
add_component(6.2, 9.1, 'Text Chunking (create_chunks)', 4)
add_component(6.2, 8.5, 'Data Cleaning & Normalization', 4)
add_component(6.2, 7.9, 'Feature Extraction', 4)
add_component(6.2, 7.3, 'Document Formatting', 4)

# Vector Store components
add_component(11.2, 9.7, 'HuggingFace Embeddings', 4)
add_component(11.2, 9.1, 'FAISS Vector Database', 4)
add_component(11.2, 8.5, 'Similarity Search', 4)
add_component(11.2, 7.9, 'Index Management', 4)
add_component(11.2, 7.3, 'Vector Serialization', 4)

# LLM System components
add_component(3.7, 5.7, 'Mistral-7B-Instruct Model', 4)
add_component(3.7, 5.1, 'Retrieval QA Chain', 4)
add_component(3.7, 4.5, 'Custom Prompt Template', 4)

# Memory System components
add_component(8.7, 5.7, 'Memory Store', 4)
add_component(8.7, 5.1, 'Add Memory Function', 4)
add_component(8.7, 4.5, 'Retrieve Memory Function', 4)

# User Interface components
add_component(6.2, 2.7, 'Streamlit Web Interface', 4)
add_component(6.2, 2.1, 'Chat Input/Output', 4)
add_component(6.2, 1.5, 'Session Management', 4)

# Draw arrows connecting systems
# Data flow
draw_arrow(5, 9.25, 6, 9.25, 'blue', '->', 2)  # Data Sources to Processing
draw_arrow(10, 9.25, 11, 9.25, 'blue', '->', 2)  # Processing to Vector Store

# LLM connections
draw_arrow(11, 8, 7.5, 5.5, 'green', '->', 2)  # Vector Store to LLM (retrieval)
draw_arrow(7.5, 5.1, 5.5, 5.1, 'green', '->', 2)  # Memory to LLM

# User interface connections
draw_arrow(5.5, 4.5, 7, 3, 'red', '->', 2)  # LLM to UI
draw_arrow(9.5, 4.5, 8, 3, 'purple', '->', 2)  # Memory to UI
draw_arrow(8, 1, 10.5, 4.5, 'purple', '->', 2)  # UI to Memory

# Set plot limits
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Add title
ax.set_title('Mental Health Support System - Architecture Overview', fontsize=18, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    patches.Patch(facecolor='#ffcccc', edgecolor='black', alpha=0.7, label='Data Sources'),
    patches.Patch(facecolor='#ccffcc', edgecolor='black', alpha=0.7, label='Data Processing'),
    patches.Patch(facecolor='#ccccff', edgecolor='black', alpha=0.7, label='Vector Storage'),
    patches.Patch(facecolor='#ffffcc', edgecolor='black', alpha=0.7, label='LLM System'),
    patches.Patch(facecolor='#ffccff', edgecolor='black', alpha=0.7, label='Memory System'),
    patches.Patch(facecolor='#ccffff', edgecolor='black', alpha=0.7, label='User Interface')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
          fancybox=True, shadow=True, ncol=3)

# Add data flow explanation
ax.text(1, 0.5, 'Data Flow:', fontsize=10, fontweight='bold')
ax.text(2.2, 0.5, 'User Query → LLM System → Vector Retrieval → Context → Response Generation → User Interface', 
        fontsize=9)

# Save the figure
plt.tight_layout()
plt.savefig('visualization_output/system_architecture.png', dpi=300, bbox_inches='tight')
print("System architecture diagram saved to 'visualization_output/system_architecture.png'")
plt.close()
