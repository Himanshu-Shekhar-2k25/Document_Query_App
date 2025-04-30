import os
from dotenv import load_dotenv
import google.generativeai as generativeai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
import streamlit as st
import shutil
from groq import Groq

uploads_location = "user_uploads"
if os.path.exists(uploads_location) and os.path.isdir(uploads_location):
    shutil.rmtree(uploads_location) 
os.makedirs(uploads_location, exist_ok=True)

def load_document(pdf_docs):
    file_path = os.path.join(uploads_location, pdf_docs.name)
    with open(file_path, "wb") as f:
        f.write(pdf_docs.getbuffer())
    file_type = pdf_docs.type
    
    if file_type == "application/pdf":
        loader = PyPDFLoader(file_path)
        pdf_doc = loader.load()
        return pdf_doc
    elif file_type == "text/plain":
        loader = TextLoader(file_path)
        txt_doc = loader.load()
        return txt_doc
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = UnstructuredWordDocumentLoader(file_path)
        docx_doc = loader.load()
        return docx_doc

def create_chunks(pdf_doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    pdf_chunks = text_splitter.split_documents(pdf_doc)
    chunk_data = [chunk.page_content for chunk in pdf_chunks]
    return chunk_data

def get_embeddings(chunk_data):
    embeddings = []
    for i, chunk in enumerate(chunk_data):
        result = generativeai.embed_content(model="models/text-embedding-004", content=chunk)
        embeddings.append({'text': chunk, 'vector': result['embedding']})
    return embeddings

def pinecone_store(embeddings, doc_name):
    records = []
    for embedding in embeddings:
        records.append({
            "id": str(st.session_state.id),
            "values": embedding['vector'],
            "metadata": {'text': embedding['text'], "filename": doc_name}})
        st.session_state.id += 1
    st.session_state.index.upsert(vectors=records, namespace="ProjectVectorStore")
    return st.session_state.index

def get_answer(query):
    query_embedding = generativeai.embed_content(model="models/text-embedding-004", content=query)['embedding']
    
    query_result = st.session_state.index.query(
        vector=query_embedding,  
        top_k=10,  
        namespace="ProjectVectorStore",
        include_metadata=True)

    context = ""
    doc_info = {}
    for match in query_result['matches']:
        current_doc = match['metadata']['filename']
        if current_doc not in doc_info:
            doc_info[current_doc] = match['metadata']['text']
        else:  
            doc_info[current_doc] += '\n' + match['metadata']['text']
    
    for key, value in doc_info.items():
        context += f"**** {key} ****\n\n{value}\n\n\n"

 
    conversation_context = "\n\n"
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        for entry in st.session_state.conversation_history[-10:]:  
            conversation_context += f"""User Query: {entry['query']}"""+'\n'+f"""Model Response: {entry['response']}\n\n"""
        conversation_context += "\n"

    prompt = """The user will ask a question or provide some information. You have three sources of information:

    1) Document Context – Text from one or more documents, provided below with filenames.

    2) User Query – The question the user asked, which can also contain additional information.

    3) Conversation History – Previous interactions that may contain relevant context or facts shared earlier.

    Carefully go through the context from each document and refer to both the User Query and Conversation History when forming your answer. If you find an answer in a document, include it in your response. For each answer from a document, start with the document name (highlighted clearly) and then provide the relevant answer. Do not repeat the same filename again. Frame answers from different documents in different paragraphs.

    Try to find an answer from as many documents as possible without misinterpreting the context. Answer only from the given Document Context and the Conversation History. Do not answer from your own knowledge or make up an answer.

    If the answer is not present in any of the documents or conversation history, reply with: "I DO NOT KNOW."

    However, if the user provides any information in the query, do not respond with "I DO NOT KNOW." Instead, incorporate the provided information and use it as part of your response if needed. This includes cases where the user asks you to remember something — treat such cases as additional context in the User Query.

    Document Context:
    {context}
    \n\n
    Conversation History:
    {conversation_context}
    \n\n
    User Query:
    {query}"""

    final_query = prompt.format(context=context, conversation_context=conversation_context, query=query)
    response = st.session_state.model.generate_content(final_query)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    st.session_state.conversation_history.append({'query': query, 'response': response.text})

    return response.text

def main():
    load_dotenv()
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    generativeai.configure(api_key=os.environ["GEMINI_API_KEY"])

    st.session_state.model = generativeai.GenerativeModel("gemini-2.0-flash-lite")
    st.session_state.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    st.session_state.index = st.session_state.pc.Index("project-index")
    st.session_state.id = 0
    st.session_state.groq_client = Groq()

    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload your documents here", type=["pdf", "txt", "docx"], accept_multiple_files=True)

        if pdf_docs is not None:
            if st.button("Process document"):
                i = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                for doc in pdf_docs:
                    status_text.text(f"Creating Embeddings for {doc.name}        {i+1}/{len(pdf_docs)}")
                    pdf_doc = load_document(doc)
                    chunk_data = create_chunks(pdf_doc)
                    embeddings = get_embeddings(chunk_data)
                    st.session_state.index = pinecone_store(embeddings, doc.name)
                    progress = (i + 1) / len(pdf_docs)
                    progress_bar.progress(progress)
                    i += 1
                status_text.text("Embeddings created successfully.")
        else:
            st.error("Please upload at least one file...")

    st.title("Document Query App")  

    choice = st.pills("Select Input Method", ("Text", "Audio"))
    query = ""
    if choice == "Text":
        query = st.text_input(label="Enter your query")
    elif choice == "Audio":
        audio_query = st.audio_input("Ask your query")
        if audio_query:
            with open("user_query.wav", "wb") as f:
                f.write(audio_query.getbuffer())
            with open("user_query.wav", "rb") as file:
                transcription = st.session_state.groq_client.audio.transcriptions.create(file=file, model="whisper-large-v3-turbo")
                query = st.text_input(label="The entered query", value=transcription.text)

    get_llm_answer = st.button("Get Answer")
    if get_llm_answer and query:
        answer = get_answer(query)
        st.write(answer)

if __name__ == '__main__':
    main()