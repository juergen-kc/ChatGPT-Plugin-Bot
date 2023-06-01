import streamlit as st
from streamlit.chat import message
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import logging

logging.basicConfig(filename="app.log", level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_data(index_path, store_path):
    """Load FAISS index and vector store for document retrieval."""
    try:
        index = faiss.read_index(index_path)
        with open(store_path, "rb") as f:
            store = pickle.load(f)
        store.index = index
        logging.info("Data loaded successfully.")
        return store
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def setup_chat_chain(vector_store):
    # ... setup_chat_chain function ...

# Load the data, set up the chain
vector_store = load_data("docs.index", "faiss_store.pkl")

if vector_store is not None:
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)

st.set_page_config(page_title="ChatGPT Plugin Bot", page_icon=":robot:")

st.header("ChatGPT Plugin Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

@st.exception
def process_input(user_input):
    conversation_history = list(zip(st.session_state["past"], st.session_state["generated"]))[-6:]
    history = '\n'.join([f'{msg["role"]}: {msg["content"]}' for msg_pair in conversation_history for msg in msg_pair])
    result = chain(history)
    return result

user_input = st.text_input("You: ", "Hello, how are you?", key="input")

if user_input:
    try:
        result = process_input(user_input)
        output = f"Answer: {result['answer']}\nSources: {result['sources']}"
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    except Exception as e:
        st.warning("An error occurred while processing your input. Please try again.")
        logging.error(f"Error processing query: {e}")

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
