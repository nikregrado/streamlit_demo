

"""# LangChain prompt templates

"""
import os
import json

import nltk

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader

nltk.download('averaged_perceptron_tagger')

with open('secrets.json', 'r') as secrets:
    secrets = json.load(secrets)
os.environ["OPENAI_API_KEY"] = secrets['OPENAI_API_KEY'] # pass your key


def create_loader():
    loader = DirectoryLoader('essays/', glob='*.txt')
    return loader


# !pip install unstructured

# https://towardsdatascience.com/let-us-extract-some-topics-from-text-data-part-iv-bertopic-46ddf3c91622
def split_documents():
    loader = create_loader()
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def vectordbqa(query: str):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])  
    texts = split_documents()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

    result = qa({"query": query})
    return result
