import os
from dotenv import load_dotenv
import google.generativeai as generativeai
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
import streamlit as st

uploads_location = "user_uploads"
os.makedirs(uploads_location, exist_ok=True)
if 'filename_dict' not in st.session_state:
    st.session_state.filename_dict = {}

def load_document(pdf_docs):
    file_path = os.path.join(uploads_location, f"user_doc{st.session_state.filename_dict[pdf_docs.name]}.pdf")

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


def pinecone_store(embeddings, i):
    records = []
    id=0
    for embedding in embeddings:
        records.append({
        "id": str(id),
        "values": embedding['vector'],
        "metadata":{'text':embedding['text']}})
        id = id+1

    stats = st.session_state.index.describe_index_stats()
    if "namespaces" in stats and f"ProjectVectorStore{i}" in stats["namespaces"]:
        st.session_state.index.delete(delete_all=True, namespace=f"ProjectVectorStore{i}")

    st.session_state.index.upsert(
        vectors=records,
        namespace=f"ProjectVectorStore{i}")
    
    return st.session_state.index

def get_answer(query, selected_document):
    query_embedding =   generativeai.embed_content(model="models/text-embedding-004",content=query)['embedding']
    
    query_result = st.session_state.index.query(
    vector=query_embedding,  
    top_k=5,  
    namespace=f"ProjectVectorStore{st.session_state.filename_dict[selected_document]}",
    include_metadata=True)


    context = ""
    for match in query_result['matches']:
        information = match['metadata']['text']
        context = context + information + '\n\n'


    prompt = "The user will ask some question. The context of the information will be provided. Answer from the given context only. DO NOT ANSWER FROM YOUR KNOWLEDGE OR TRY TO MAKE UP SOME ANSWER. IF ANSWER IS NOT PRESENT REPLY 'I DO NOT KNOW'"

    final_query = prompt + '\n\n' + context +'\n\n' + query


    response = st.session_state.client.models.generate_content(
    model="gemini-1.5-flash", contents=final_query)

    return response.text

def main():
    load_dotenv()
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    generativeai.configure(api_key=os.environ["GEMINI_API_KEY"])

    st.session_state.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    st.session_state.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    st.session_state.index = index = st.session_state.pc.Index("project-index")


    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload your PDFs here", type=["pdf"], accept_multiple_files=True)


        if pdf_docs is not None and len(pdf_docs) <= 5:
            if st.button("Process document"):
                # st.session_state.filename_dict = {}
                i = 0
                progress_bar = st.progress(0)
                status_text = st.empty()

                for doc in pdf_docs:
                    status_text.text(f"Creating Embeddings for {doc.name}        {i+1}/{len(pdf_docs)}")
                    st.session_state.filename_dict[doc.name] = i

                    pdf_doc = load_document(doc)
                    chunk_data = create_chunks(pdf_doc)
                    embeddings = get_embeddings(chunk_data)
                    st.session_state.index = pinecone_store(embeddings, i)

                    progress = (i + 1) / len(pdf_docs)
                    progress_bar.progress(progress)
                    
                    i = i+1
                status_text.text("Embeddings created succesfully.")
        else:
            st.error("Max 5 documents can be uploaded.")
        

    st.title("Document Query App")  

    selected_document = st.selectbox("Choose a file to query", [file for file in st.session_state.filename_dict])
    query = st.text_input(label="Enter your query")
    get_llm_answer = st.button("Get Answer")
    if(get_llm_answer):
        answer = get_answer(query, selected_document)
        st.write(answer)
    

if __name__ == '__main__':
    main()