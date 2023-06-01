from pathlib import Path
import csv
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

# Prepare list of all CSV file paths
file_paths = list(Path("Data/").glob("**/*.csv"))

docs = []
metadatas = []

# Initialize the text splitter with desired chunk size
text_splitter = CharacterTextSplitter(chunk_size=1900, separator="\n")

for file_path in file_paths:
    try:
        # Use 'with' statement to ensure file is properly closed after being used
        with open(file_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            # Process each line in the CSV
            for row in csv_reader:
                # Concatenate column values into a single string, prefixed with their column names
                data = ' '.join([f'{k}: {v}' for k, v in row.items()])
                # Split the data into chunks
                splits = text_splitter.split_text(data)
                docs.extend(splits)
                # Associate each chunk with its source file
                metadatas.extend([{"source": str(file_path)}] * len(splits))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

# Create a vector store from the documents
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)

# Save the index separately
faiss.write_index(store.index, "docs.index")

# Save the store without the index
store.index = None
with open("faiss_store.pkl", "wb") as f:
  pickle.dump(store, f)
