import json
import os
import pickle
import logging
import faiss
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from debug_chains import DebugRetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, template_folder='templates')
CORS(app)  # handle Cross origin resource sharing

def load_data(index_path, store_path):
    try:
        index = faiss.read_index(index_path)
        with open(store_path, "rb") as f:
            vector_store = pickle.load(f)
        vector_store.index = index
        logging.info("Data loaded successfully.")
        return vector_store
    except Exception:
        logging.exception("Error loading data")
        return None

def setup_chat_chain(vector_store):
    if vector_store is None:
        return None
    
    system_template ="""You are an AI assitant that provides information about ChatGPT Plugins available in the Plugin Store. 
You have access to data for each Plugin that includes 'Plugin Name', 'Plugin Description', 'End User Instructions', as well as 16 plugins that are ranked as Popular. 
When asked questions, search for match in the 'Plugin Name' and 'Plugin Description' of all plugins. 
If there are multiple plugins that may be related to the question, please include all of them in your answer
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256)
    chain = DebugRetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        debug=True
    )
    logging.info("Chat chain setup completed.")
    return chain

vector_store = load_data("docs.index", "faiss_store.pkl")
chain = setup_chat_chain(vector_store)

@app.route("/ask", methods=['POST'])
def ask():
    if chain is None:
        return jsonify({'error': 'Failed to load data. Please check the log for details.'}), 500
    question = request.get_json(force=True)
    query = question["query"]
    try:
        result = chain(query)
        return jsonify({'answer': result['answer']})
    except Exception:
        logging.exception("Error processing query")
        return jsonify({'error': 'Server error'}), 500

@app.route('/')
def chat_page():
    return render_template('chat.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
