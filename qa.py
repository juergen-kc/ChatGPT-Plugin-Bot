import faiss
import logging
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from debug_chains import DebugRetrievalQAWithSourcesChain
import pickle
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup the logging configuration
# Logging is useful for tracking the application's activities and catching any unexpected behaviors.
# We're setting the level to DEBUG meaning it will capture all messages down to the most granular level.
# These messages will be written to a file named 'app.log' in the same directory as the script.
logging.basicConfig(filename="app.log", level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_data(index_path, store_path):
    """Load FAISS index and vector store for document retrieval."""
    try:
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load vector store
        with open(store_path, "rb") as f:
            vector_store = pickle.load(f)
        
        # Reattach the FAISS index to the vector store
        vector_store.index = index

        logging.info("Data loaded successfully.")
        return vector_store
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


def setup_chat_chain(vector_store):
    """Setup the chat model and retrieval chain."""
    # System message template
    system_template="""You are an AI assitant that provides information about ChatGPT Plugins available in the Plugin Store. 
You have access to data for each Plugin that includes 'Plugin Name', 'Plugin Description', 'End User Instructions', as well as 16 plugins that are ranked as Popular. 
When asked questions, search for match in the 'Plugin Name' and 'Plugin Description' of all plugins. 
If there are multiple plugins that may be related to the question, please include all of them in your answer
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""""
    
    # Message templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    
    # Compile chat prompt
    prompt = ChatPromptTemplate.from_messages(messages)
    
    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256)
    
    # Create the retrieval chain
    chain = DebugRetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    debug=True  # set to False if you want to turn off the logging
    )
    
    logging.info("Chat chain setup completed.")
    return chain


def print_result(query, result):
  """Print the chat interaction results in a clean format."""
  output_text = f"""### Question: 
  {query}
  ### Answer: 
  {result['answer']}
  """
  print(output_text)


def chat_loop(chain):
    """Main chat loop where user inputs are processed and responses are generated."""
    conversation_history = []
    while True:
        try:
            query = input("Please enter your question or type 'quit' to exit: ")
            if query.lower() == 'quit':
                break

            # Append user's query to the conversation history
            conversation_history.append({"role": "user", "content": query})

            # Prepare the prompt with conversation history
            history = '\n'.join([f'{msg["role"]}: {msg["content"]}' for msg in conversation_history[-6:]]) # Keep only the last 6 messages

            result = chain(history)

            # Append assistant's response to the conversation history
            conversation_history.append({"role": "assistant", "content": result['answer']})

            print_result(query, result)
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            print("An error occurred. Please try again.")


# Load the data, set up the chain, and start the chat loop
vector_store = load_data("docs.index", "faiss_store.pkl")
if vector_store is not None:
    chain = setup_chat_chain(vector_store)
    chat_loop(chain)
else:
    print("Failed to load data. Please check the log for details.")
