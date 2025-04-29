import os
import streamlit as st
import uuid
from pinecone import Pinecone
from dotenv import load_dotenv
import sys
import importlib.util

# Check if protobuf is installed
try:
    from google.protobuf import descriptor as _descriptor
except ImportError:
    st.error("""
    ## Missing dependency: google-protobuf
    
    Streamlit requires the protobuf library to function properly. Please install it using:
    ```
    pip install protobuf
    ```
    
    Then restart this app.
    """)
    st.stop()

# Add the current directory to the path
from search_similar import search_similar_texts
from insert_text import insert_text

# Load environment variables from .env in the current directory
load_dotenv(dotenv_path=".env")

# Set page configuration
st.set_page_config(
    page_title="Pinecone Text Search & Insert",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextArea label {
        font-weight: 600;
    }
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Get default values from environment variables
default_api_key = os.getenv("PINECONE_API_KEY", "")
default_index_name = os.getenv("PINECONE_INDEX_NAME", "msft-deep-search-oai-3-small")
default_namespace = os.getenv("PINECONE_NAMESPACE", "")
default_openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Sidebar for configuration
with st.sidebar:
    st.title("🔍 Configuration")
    
    # Using expander for API settings
    with st.expander("API Settings", expanded=True):
        # API Key
        api_key = st.text_input(
            "Pinecone API Key", 
            value=default_api_key, 
            type="password", 
            help="Enter your Pinecone API Key",
            placeholder="Enter API key here..."
        )
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            value=default_openai_api_key, 
            type="password", 
            help="Enter your OpenAI API Key",
            placeholder="Enter OpenAI API key here..."
        )
        # Index name input
        index_name = st.text_input(
            "Index Name", 
            value=default_index_name, 
            help="Name of the Pinecone index",
            placeholder="Enter index name..."
        )
        # Namespace input
        namespace = st.text_input(
            "Namespace", 
            value=default_namespace, 
            help="Namespace within the index (leave empty for default)",
            placeholder="Optional namespace..."
        )
    st.divider()
    st.info("💡 This app lets you search for similar texts and insert new texts into Pinecone vector database using OpenAI embeddings.")
    st.caption("Made with Streamlit, OpenAI, and Pinecone")

# Main content
st.title("Pinecone Text Vectorstore App")

# Create tabs for different functions with icons
tab1, tab2 = st.tabs(["🔍 Search Similar Texts", "➕ Insert Text"])

# Search tab
with tab1:
    st.header("Search for Similar Texts")
    
    # Query input with clear messaging
    query = st.text_area(
        "Enter your search query",
        height=150,
        placeholder="Type your search query here..."
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Number of results to return with a more intuitive slider
        top_k = st.slider(
            "Number of results to return",
            min_value=1,
            max_value=50,
            value=5,
            help="Slide to adjust how many similar results to fetch"
        )
    
    with col2:
        # Vertical centering with a spacer
        st.markdown("<br>", unsafe_allow_html=True)
        # Search button
        search_clicked = st.button(
            "🔎 Search", 
            key="search_button",
            type="primary",
            use_container_width=True
        )
    
    if search_clicked:
        if not query:
            st.error("⚠️ Please enter a search query")
        elif not api_key:
            st.error("⚠️ Please enter your Pinecone API Key in the sidebar")
        elif not openai_api_key:
            st.error("⚠️ Please enter your OpenAI API Key in the sidebar")
        elif not index_name:
            st.error("⚠️ Please enter an index name in the sidebar")
        else:
            try:
                with st.status("Searching...", expanded=True) as status:
                    st.write("Creating embedding for your query using OpenAI...")
                    results = search_similar_texts(
                        query, 
                        top_k=top_k, 
                        api_key=api_key,
                        index_name=index_name,
                        namespace=namespace,
                        openai_api_key=openai_api_key
                    )
                    st.write("Search complete!")
                    status.update(label="Search complete!", state="complete")
                
                if results:
                    st.success(f"✅ Found {len(results)} similar texts")
                    
                    # Add information about semantic similarity scores
                    with st.expander("ℹ️ About Semantic Similarity Scores", expanded=True):
                        st.markdown("""
                        ### Semantic Similarity Explained
                        
                        Semantic similarity measures how close two pieces of text are in **meaning**, regardless of their exact wording.
                        
                        #### What the scores mean:
                        - **0.8 - 1.0**: Very high similarity (green) - nearly identical in meaning
                        - **0.6 - 0.8**: Good similarity (orange) - clearly related content
                        - **0.0 - 0.6**: Lower similarity (red) - some related concepts but different focus
                        
                        #### How it works:
                        1. Your query is converted to a vector using the SentenceTransformer model
                        2. This vector is compared to all text vectors in the Pinecone database
                        3. The cosine similarity between vectors produces the similarity score
                        4. Higher scores indicate greater semantic relevance to your query
                        
                        _The score is a measure of conceptual similarity, not just keyword matching._
                        """)
                    
                    # Visualization of similarity scores
                    st.subheader("📊 Semantic Similarity Distribution")
                    st.caption("Visual representation of how similar each result is to your query")
                    
                    # Extract scores for visualization
                    scores = [item['similarity_score'] for item in results]
                    max_score = max(scores) if scores else 0
                    
                    # Create columns for the visualization
                    viz_cols = st.columns(len(results))
                    
                    # Display progress bars for each result
                    for i, (col, score) in enumerate(zip(viz_cols, scores)):
                        with col:
                            st.markdown(f"**#{i+1}**")
                            # Use progress bar to visualize score - Normalize to 0-1 range if needed
                            normalized_score = min(score, 1.0)  # Cap at 1.0 for the progress bar
                            st.progress(normalized_score)
                            st.markdown(f"<div style='text-align: center;'>{score:.3f}</div>", unsafe_allow_html=True)
                    
                    # Add a horizontal line to separate visualization from results
                    st.markdown("---")
                    
                    # Create result display cards
                    for i, item in enumerate(results, 1):
                        with st.container():
                            st.divider()
                            
                            # Display score prominently with better explanation
                            sim_score = item['similarity_score']
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.subheader(f"Result {i}")
                            with col2:
                                # Use color coding based on similarity score
                                score_color = "green" if sim_score > 0.8 else "orange" if sim_score > 0.6 else "red"
                                st.markdown(f"""
                                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;">
                                    <span style="font-size: 0.8em; color: gray;">Semantic Similarity</span><br/>
                                    <span style="font-size: 1.5em; font-weight: bold; color: {score_color};">{sim_score:.4f}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show the text content
                            st.text_area(
                                label="",
                                value=item['text'], 
                                height=150,
                                key=f"result_{i}",
                                disabled=True
                            )
                            
                            # Use columns for metadata and id
                            mcol1, mcol2 = st.columns([3, 1])
                            
                            with mcol1:
                                # Display all metadata except the text
                                meta_display = {k: v for k, v in item['metadata'].items() if k != 'text'}
                                if meta_display:
                                    with st.expander("Additional Metadata"):
                                        st.json(meta_display)
                            
                            with mcol2:
                                st.caption(f"ID: {item['id']}")
                else:
                    st.info("ℹ️ No results found")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Insert Text tab
with tab2:
    st.header("Insert Text into Pinecone")
    
    # Text input with clearer instruction
    text = st.text_area(
        "Enter the text to insert",
        height=200,
        placeholder="Type the text you want to add to the vector database here..."
    )
    
    # Metadata inputs section with better structure
    st.subheader("Additional Metadata (Optional)")
    
    # Dynamic metadata fields
    metadata = {}
    
    # Better layout for metadata fields
    for i in range(3):  # Allow up to 3 metadata fields for cleaner UI
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                key = st.text_input(
                    "Key", 
                    key=f"meta_key_{i}", 
                    placeholder="e.g., category, author, date...",
                    label_visibility="collapsed"
                )
            with col2:
                value = st.text_input(
                    "Value", 
                    key=f"meta_value_{i}", 
                    placeholder="Value",
                    label_visibility="collapsed"
                )
            
            if key and value:
                metadata[key] = value
    
    # Insert button with better positioning
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        insert_clicked = st.button(
            "➕ Insert Text", 
            key="insert_button",
            type="primary",
            use_container_width=True
        )
    
    if insert_clicked:
        if not text:
            st.error("⚠️ Please enter text to insert")
        elif not api_key:
            st.error("⚠️ Please enter your Pinecone API Key in the sidebar")
        elif not openai_api_key:
            st.error("⚠️ Please enter your OpenAI API Key in the sidebar")
        elif not index_name:
            st.error("⚠️ Please enter an index name in the sidebar")
        else:
            try:
                with st.status("Processing...", expanded=True) as status:
                    st.write("Creating embedding for your text using OpenAI...")
                    vector_id = insert_text(
                        text, 
                        metadata=metadata, 
                        api_key=api_key,
                        index_name=index_name,
                        namespace=namespace,
                        openai_api_key=openai_api_key
                    )
                    st.write("Saving to Pinecone...")
                    status.update(label="Insertion complete!", state="complete")
                
                # Show success message with animated confetti
                st.success("✅ Successfully inserted text into Pinecone")
                st.balloons()
                
                # Display results in a more visually appealing way
                st.write("##### Vector Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Vector ID**: {vector_id}")
                with col2:
                    st.info(f"**Namespace**: {namespace or 'default'}")
                
                st.info(f"**Index**: {index_name}")
                
                if metadata:
                    st.write("##### Added Metadata")
                    st.json(metadata)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e) 