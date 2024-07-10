import os
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from constants import PINECONE_INDEX_NAME
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import pinecone

# Set API key for OpenAI and Pinecone
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENVIRONMENT
)

# First, check if our index already exists. If it doesn't, we create it
print('Checking if index exists...')
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    print('Index does not exist, creating index...')
    # we create a new index
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        metric='cosine',
        # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
        dimension=1536
    )

print('Loading document...')
# Load PDF
loader = UnstructuredPDFLoader(
    "Medical TextBook".pdf')

data = loader.load()

print(f'You have loaded a PDF with {len(data)} pages')
print(f'There are {len(data[0].page_content)} characters in your document')

# Chunk data into smaller documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)

print(f'You have split your document into {len(texts)} smaller documents')

# Create embeddings and index from your documents
print('Creating embeddings and index...')
embeddings = OpenAIEmbeddings(client='')
docsearch = Pinecone.from_texts(
    [t.page_content for t in texts], embeddings, index_name=PINECONE_INDEX_NAME)

print('Embeddings and indexing complete!')

# Set up the LLM for retrieval and QA
llm = OpenAI(temperature=0.7, max_tokens=150)

# Create the RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(),
    chain_type="stuff"
)

def get_answer(query):
    print(f'Retrieving relevant documents for the query: "{query}"')
    answer = retrieval_qa.run(query)
    return answer

# Example usage
query = "Explain about the Skin Diseases?"
answer = get_answer(query)
print(f'Answer: {answer}')

query = "What is the Tablet used to cure headache?"
answer = get_answer(query)
print(f'Answer: {answer}')


query = "What are the Symptoms of fever?"
answer = get_answer(query)
print(f'Answer: {answer}')
