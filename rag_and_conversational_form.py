import os 
from FileParser.fileparser import FileParserFactory
from RAG_System.vector_store_maker import VectorStoreMakingFactory

from config import GEMINI_API_KEY,HUGGINGFACE_API_KEY,COHERE_RERANK_API_KEY,DATABASE_URL,MODEL_NAME
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough

from prompt.prompt import get_prompt
from source.chain import get_chain

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from dateutil import parser
from datetime import datetime, timedelta
import spacy
from langchain.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field
from typing import List, Type, Union, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
import random
import warnings
warnings.filterwarnings("ignore")



from conversationa_form import UserInfoCollectorWithToolAndAgent
LLM_TYPE="gemini"
used_api_key=GEMINI_API_KEY


embeddings=HuggingFaceInferenceAPIEmbeddings(
api_key=HUGGINGFACE_API_KEY,
model_name='BAAI/bge-base-en-v1.5'
)

def vector_store_creator_from_file(file_name,embeddings,splitting_type: str = "recursive"):
    file_location=file_name
    file_extension = os.path.splitext(file_location)[1][1:]
    file_parser=FileParserFactory(file_type=file_extension,file_name=file_location)
    content=file_parser.parse()

    vector_store_maker=VectorStoreMakingFactory(splitting_type=splitting_type,document=content,file_extension=file_extension)
    vector_store_docs=vector_store_maker.splittext()
   
    vectorstore = FAISS.from_documents(vector_store_docs, embeddings)
    
    return vector_store_docs,vectorstore

def retriever_maker_for_rag_and_compressor(vectorstore,vector_store_docs):
    retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(vector_store_docs)
    keyword_retriever.k =  5
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.7, 0.3])
    compressor = CohereRerank(cohere_api_key=COHERE_RERANK_API_KEY)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever)
    return compression_retriever

def qa_chain_maker(api_key,model_name,compression_retriever):
    template = """
    <|system|>>
    You are an AI Assistant that follows instructions extremely well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

    CONTEXT: {context}

    <|user|>
    {query}

    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    llm = ChatGoogleGenerativeAI(api_key=api_key, temperature=0, model=model_name)


    qa_chain = (
        {"context": compression_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return qa_chain

def get_classification_llm_chain(model,used_api_key):
    CHAT_TYPE_PROMPT_LOC = r"C:\Users\prabigya\Desktop\work_here\chatbot_docs_form\prompt\classify_query.tmpl"
    prompt = get_prompt(
                path=CHAT_TYPE_PROMPT_LOC,
                vars={
                    "input": "{input}"
                },
            )
    chain_i =get_chain(LLM_TYPE=LLM_TYPE,api_key=used_api_key,temperature=0.1, model=model, prompt=prompt)
    chain=chain_i|StrOutputParser()
    return chain


def classify_user_query(input_text, chain):
        appointment_triggers = ['call me', 'book me', 'schedule for','schedule me']
        if any(trigger in input_text.lower() for trigger in appointment_triggers):
            return "Appointment"       
        response = chain.invoke({"input": input_text})
        return response


def handle_user_input(user_query,classification_chain,qa_chain):
    classification=classify_user_query(user_query,classification_chain)
    if str(classification).lower().strip() != "appointment":
        result=qa_chain.invoke(user_query)
        return result

    else:
        user_info_with_tool_agent=UserInfoCollectorWithToolAndAgent(api_key=GEMINI_API_KEY,model=MODEL_NAME)
        user_info_with_tool_agent.collect_user_information(user_query)
        return user_info_with_tool_agent.user_info
    
api_key=GEMINI_API_KEY
model_name=MODEL_NAME

while(1):
    file_location=r"C:\Users\prabigya\Desktop\work_here\chatbot_docs_form\FileParser\sample2.json"
    vector_store_docs,vectorstore=vector_store_creator_from_file(file_location,embeddings=embeddings,splitting_type="recursive")
    compression_retriever=retriever_maker_for_rag_and_compressor(vectorstore=vectorstore,vector_store_docs=vector_store_docs)

    qa_chain=qa_chain_maker(api_key=api_key,model_name=model_name,compression_retriever=compression_retriever)
    classification_chain=get_classification_llm_chain(model=model_name,used_api_key=api_key)
    query=str(input("Enter Question: "))
    if query.lower().strip() == "exit":
        break
    if len(query.strip())>0:
        response=handle_user_input(query,classification_chain,qa_chain)
        print(response)

