# Build-a-Knowledge-Based-System-with-Vertex-AI-Vector-Search-LangChain-and-Gemini

1. Installation and Setup

%%capture --no-stderr
!pip3 install -q --upgrade pip
!pip3 install -q google-cloud-aiplatform
!pip3 install -q langchain
!pip3 install -q langchain-community
!pip3 install -q lxml
!pip3 install -q requests
!pip3 install -q beautifulsoup4
!pip3 install -q unstructured
!pip3 install -q langchain-google-genai
!pip3 install -q google-generativeai
!pip3 install -q tqdm

# restart the kernel
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

# Purpose: 

This section installs the necessary Python packages, including Google Cloud AI Platform, LangChain, and other dependencies such as BeautifulSoup for web scraping, and tqdm for progress bars. After installing the packages, the kernel is restarted to ensure that the newly installed packages are properly loaded.

2. Initial Setup

from IPython.display import display
from IPython.display import Markdown
import textwrap

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# source API key from GCP project and configure genai client

import os
import pathlib
import textwrap
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

key_name = !gcloud services api-keys list --filter="gemini-api-key" --format="value(name)"
key_name = key_name[0]

api_key = !gcloud services api-keys get-key-string $key_name --location="us-central1" --format="value(keyString)"
api_key = api_key[0]

os.environ["GOOGLE_API_KEY"] = api_key

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Purpose: 

This section sets up the environment by displaying Markdown text, sourcing the API key from Google Cloud, and configuring the genai client with the API key. The to_markdown function formats text as Markdown for display purposes.

3. Define Project Information

PROJECT_ID = subprocess.check_output(["gcloud", "config", "get-value", "project"], text=True).strip()
REGION = "us-central1"  # @param {type:"string"}

print(f"Your project ID is: {PROJECT_ID}")

BUCKET = f"gs://{PROJECT_ID}/embeddings"
DIMENSIONS=768
DISPLAY_NAME='vertex_docs_qa'
ENDPOINT=f"{REGION}-aiplatform.googleapis.com"
TEXT_GENERATION_MODEL='gemini-pro'
SITEMAP='https://docs.anthropic.com/sitemap.xml'
Purpose: This part defines key project variables such as PROJECT_ID, REGION, BUCKET, DIMENSIONS, and others. These variables will be used throughout the notebook to manage data storage, embedding dimensions, and model parameters.

4. Load and Parse the Sitemap

import requests
from bs4 import BeautifulSoup

def parse_sitemap(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    urls = [element.text for element in soup.find_all("loc")]
    return urls

sites = parse_sitemap(SITEMAP)

# Use this to filter out docs that don't have a corresponding reference page

sites_filtered = [url for url in sites if '/en/docs' in url]
len(sites_filtered)

# Purpose:

This code parses the sitemap XML file to extract URLs of the documentation pages. It filters out URLs that do not match the criteria (in this case, pages that don't contain /en/docs). This prepares the URLs for further processing.

5. Load Documentation Pages

from langchain.document_loaders import UnstructuredURLLoader
loader = UnstructuredURLLoader(urls=sites_filtered)
documents = loader.load();

# Purpose: 

The UnstructuredURLLoader from LangChain is used to load the content from the filtered URLs. The documents are then stored in the documents variable, which contains the content of the pages.

# 6. Create Document Chunks

import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 100)

document_chunks = text_splitter.split_documents(documents)

document_chunks=[f"content: {chunk.page_content}, source: {chunk.metadata['source']}" for chunk in document_chunks]

# Purpose: 

This section splits the documents into smaller chunks of text (each chunk being 2000 characters long with an overlap of 100 characters) to create manageable pieces of data that can be embedded later. This is necessary because large texts need to be broken down into smaller, contextually relevant parts for effective embedding and retrieval.

7. Generate Embeddings

!mkdir ./documents

import pandas as pd

df = pd.DataFrame(document_chunks, columns =['text'])

from tqdm import tqdm
import json

index_embeddings = []
model = "models/embedding-001"

for index, doc in tqdm(df.iterrows(), total=len(df), position=0):

    response = genai.embed_content(model=model, content=doc['text'], task_type="retrieval_query")

    doc_id=f"{index}.txt"
    embedding_dict = {
        "id": doc_id,
        "embedding": response["embedding"],
    }
    index_embeddings.append(json.dumps(embedding_dict) + "\n")
    
    with open(f"documents/{doc_id}", "w") as document:
          document.write(doc['text'])
    
with open("embeddings.json", "w") as f:
    f.writelines(index_embeddings)

# Purpose: 

This code generates embeddings for each document chunk using the genai.embed_content function. The embeddings are saved in JSON format, and the original document text is saved in individual text files.

8. Upload Embeddings to Cloud Storage

from google.cloud import storage

source_file = '/home/jupyter/embeddings.json'
destination_blob_name = 'embeddings/embeddings.json'

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(PROJECT_ID)
blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(source_file)

subprocess.run(['gsutil', '-q', 'cp', '-r', './documents', f'gs://{PROJECT_ID}/documents'])

# Purpose: 

The embeddings and documents are uploaded to Google Cloud Storage. This is necessary for using these embeddings in Vertex AI Vector Search.

9. Create Vertex AI Vector Store Index

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
      display_name="vertex_docs",
      contents_delta_uri=f"gs://{PROJECT_ID}/embeddings",
      dimensions=768,
      approximate_neighbors_count=150,
      distance_measure_type="DOT_PRODUCT_DISTANCE"
)

# Purpose: 

This creates an index in Vertex AI Vector Search using the uploaded embeddings. The DOT_PRODUCT_DISTANCE metric is used for similarity calculations. This index allows for fast, efficient searches through the embedded document vectors.

10. Deploy the Index Endpoint

index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="vertex_docs",
    description="Embeddings for the documentation curated from the sitemap.",
    public_endpoint_enabled=True,
)

index_endpoint = index_endpoint.deploy_index(
    index=index, deployed_index_id="vertex_index_deployment"
)

# Purpose: This code deploys the index to an endpoint, making it accessible for querying. This endpoint will be used to search for similar documents based on queries.

11. Search Vector Store and Query Model

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.matching_engine import MatchingEngine

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def search_vector_store(question):
    vector_store = MatchingEngine.from_components(
                        index_id=INDEX_RESOURCE_NAME,
                        region=REGION,
                        embedding=embeddings,
                        project_id=PROJECT_ID,
                        endpoint_id=deployed_index[0].index_endpoint,
                        gcs_bucket_name=f"{PROJECT_ID}")
    
    relevant_documentation=vector_store.similarity_search(question, k=8)
    context = "\n".join([doc.page_content for doc in relevant_documentation])[:10000]
    return str(context)

# Purpose: 

This function performs a similarity search on the vector store to retrieve relevant document chunks based on a user query. The retrieved content is used as context for generating answers.

12. Use LangChain for Retrieval-Augmented Generation

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0, convert_system_message_to_human=True)

prompt = PromptTemplate(input_variables=["context",  "question"], template=template)

vector_store = MatchingEngine.from_components(
    index_id=INDEX_RESOURCE_NAME,
    region=REGION,
    embedding=embeddings,
    project_id=PROJECT_ID,
    endpoint_id=deployed_index[0].index_endpoint,
    gcs_bucket_name=f"{PROJECT_ID}"
)

retriever = vector_store.as
