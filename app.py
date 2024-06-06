from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


loader=PyPDFLoader('cp_handbook.pdf')
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)



db = Chroma.from_documents(documents,OllamaEmbeddings(model="llama3"))

question = "Describe bellman ford algorithm."
response = db.similarity_search(question)
print(response)

