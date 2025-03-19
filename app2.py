import os
from dotenv import load_dotenv
import google.generativeai as generativeai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
import streamlit as st
import shutil

uploads_location = "user_uploads"
if os.path.exists(uploads_location) and os.path.isdir(uploads_location):
    shutil.rmtree(uploads_location) 
os.makedirs(uploads_location, exist_ok=True)

def load_document(pdf_docs):
    file_path = os.path.join(uploads_location,pdf_docs.name)

    with open(file_path, "wb") as f:
        f.write(pdf_docs.getbuffer())
                
        loader = PyPDFLoader(file_path)
        pdf_doc = loader.load()
        return pdf_doc

def create_chunks(pdf_doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    pdf_chunks = text_splitter.split_documents(pdf_doc)

    chunk_data = []
    for chunk in pdf_chunks:
        chunk_data.append(chunk.page_content)
    return chunk_data

def get_embeddings(chunk_data):
    embeddings = []

    for i, chunk in enumerate(chunk_data):
        result = generativeai.embed_content(model="models/text-embedding-004",content=chunk)
        embeddings.append({'text':chunk, 'vector':result['embedding']})

    return embeddings


def pinecone_store(embeddings, doc_name):
    records = []
    for embedding in embeddings:
        records.append({
        "id": str(st.session_state.id),
        "values": embedding['vector'],
        "metadata":{'text':embedding['text'], "filename":doc_name}})
        st.session_state.id = st.session_state.id+1

    st.session_state.index.upsert(
        vectors=records,
        namespace="ProjectVectorStore")
    
    return st.session_state.index

def get_answer(query):
    query_embedding = generativeai.embed_content(model="models/text-embedding-004",content=query)['embedding']
    
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
            doc_info[current_doc] = doc_info[current_doc] + '\n' + match['metadata']['text']

    for key, value in doc_info.items():
        context = context + f"**** {key} ****" + '\n\n' + value + '\n\n'
        # information = match['metadata']['text']
        # context = context + information + '\n\n'


    prompt = """The user will ask some question. The context of the information will be provided in the form of text from one or more documents. Carefully go through the provided context from each document. If you find any suitable answer from a document include it in your answer. TRY TO FIND ANSWER FROM AS MANY DOCUMENTS AS POSSIBLE WITHOUT MISINTERPRETING THE CONTEXT. For each answer from a document, specify the document name and then the required answer from that document context. IN YOUR RESPONSE HIGHLIGHT THE FILENAME FROM WHICH YOU ARE GIVING RESPONSE AND DO NOT REPEAT THE SAME FILE NAME AGAIN.
    
    Answer from the given context only. DO NOT ANSWER FROM YOUR KNOWLEDGE OR TRY TO MAKE UP SOME ANSWER. IF ANSWER IS NOT PRESENT REPLY 'I DO NOT KNOW.'"""

    # with open("context_info.txt", "w", encoding="utf-8") as f:
    #     f.write(context)

    final_query = prompt + '\n\n' + context +'\n\n' + query
    response = st.session_state.model.generate_content(final_query)

    return response.text

def main():
    load_dotenv()
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    generativeai.configure(api_key=os.environ["GEMINI_API_KEY"])

    st.session_state.model = generativeai.GenerativeModel("gemini-2.0-flash")
    st.session_state.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    st.session_state.index = st.session_state.pc.Index("project-index")
    st.session_state.id = 0

    if 'pinecone_preprocess' not in st.session_state:
        stats = st.session_state.index.describe_index_stats()
        if "namespaces" in stats and "ProjectVectorStore" in stats["namespaces"]:
            st.session_state.index.delete(delete_all=True, namespace="ProjectVectorStore")
        st.session_state.pinecone_preprocess = True


    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload your PDFs here", type=["pdf"], accept_multiple_files=True)


        if pdf_docs is not None:
            if st.button("Process document"):
                i = 0
                progress_bar = st.progress(0)
                status_text = st.empty()

                for doc in pdf_docs:
                    status_text.text(f"Creating Embeddings for {doc.name}        {i+1}/{len(pdf_docs)}")
                    # st.session_state.filenames.append(doc.name)

                    pdf_doc = load_document(doc)
                    chunk_data = create_chunks(pdf_doc)
                    embeddings = get_embeddings(chunk_data)
                    st.session_state.index = pinecone_store(embeddings, doc.name)

                    progress = (i + 1) / len(pdf_docs)
                    progress_bar.progress(progress)
                    
                    i = i+1
                status_text.text("Embeddings created succesfully.")
        else:
            st.error("Please upload atleast one file...")
        

    st.title("Document Query App")  

    # selected_document = st.selectbox("Choose a file to query", [file for file in st.session_state.filename_dict])
    query = st.text_input(label="Enter your query")
    get_llm_answer = st.button("Get Answer")
    if(get_llm_answer):
        answer = get_answer(query)
        st.write(answer)
    

if __name__ == '__main__':
    main()