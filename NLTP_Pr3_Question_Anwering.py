import streamlit as st
from transformers import pipeline

# Set up the page configuration
st.set_page_config(page_title="Simple Q&A System", layout="wide")

# Initialize the QA pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

qa_pipeline = load_qa_pipeline()

# Create the Streamlit interface
st.title("📚 Simple Question Answering System")
st.write("Enter your context and question below to get an answer!")

# Input fields
context = st.text_area("Context:", height=200, 
    placeholder="Paste your text context here...")
question = st.text_input("Question:", 
    placeholder="Type your question here...")

# Process input and display results
if st.button("Get Answer"):
    if context and question:
        with st.spinner("Finding the answer..."):
            try:
                result = qa_pipeline(question=question, context=context)
                
                st.success("Answer found!")
                st.write("### Answer:")
                st.write(result['answer'])
                
                st.write("### Confidence Score:")
                st.progress(result['score'])
                st.write(f"Confidence: {result['score']:.2%}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please provide both context and question!")
