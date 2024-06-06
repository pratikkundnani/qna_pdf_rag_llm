from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
## Load Ollama LAMA2 LLM model



loader=PyPDFLoader('Bodybuilding.pdf')
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Answer should be short and to the point. 
Here is the context: 
{context}
Question: {input}""")


llm=Ollama(model="llama2")
document_chain=create_stuff_documents_chain(llm,prompt)
db = FAISS.from_documents(documents,OllamaEmbeddings(model="llama3"))
retriever=db.as_retriever()

retrieval_chain=create_retrieval_chain(retriever,document_chain)
response=retrieval_chain.invoke({"input":"Surname of Arnold?"})

print(response['answer'])

