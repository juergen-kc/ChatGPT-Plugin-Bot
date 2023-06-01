# ChatGPT Custom Data Question-Answering

🤖Ask questions about any CSV dataset using ChatGPT with document retrieval. The default data set is information about ChatGPT Plugins.

💪 Built with [LangChain]

# 🌲 Environment Setup

In order to set your environment up to run the code here, first install all requirements:
```shell
pip install -r requirements.txt
```

Then set your OpenAI API key (if you don't have one, get one [here](https://beta.openai.com/signup)) as a system environment variable:

export OPENAI_API_KEY=....

# 📄 What is in here?
- Python script 'ingest.py' to embed your data
- Python script 'debug_chains.py' for setting up the Debug Retrieval Question-Answering chain
- Python script 'qa.py' for running the Question-Answering system
- Instructions for ingesting your own dataset and running the QA system

## 💬 Ask a question
In order to ask a question, run the command below and follow the prompts:
python qa.py

The prompt preserves your previous questions, and it should be able to recall what you've previously asked or the information it has shared with you.

## 🧑 Instructions for ingesting your own dataset

Export your dataset to CSV and place it in a folder named "Data" in the root directory of the project. 

Run the following command to ingest the data:
```shell
python ingest.py
```

### Boom! Now you're done, and you can ask it questions about your own data:
```shell
python qa.py
```
