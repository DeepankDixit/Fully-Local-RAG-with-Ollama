import os
import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_vectorstore_from_pdf(uploaded_docs):
    
    st.write("Uploaded files are: \n")
    for file in uploaded_docs:
        st.write(file.name)
    
    with st.sidebar:
        with st.spinner('Loading the document...'):
            # document loading
            text = ""
            for file in uploaded_docs:
                if file.name.endswith('.pdf'):
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                elif file.name.endswith('.docx'):
                    # Create a temporary file and write the uploaded file's content to it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_file_path = tmp_file.name
                    word_loader = Docx2txtLoader(tmp_file_path)
                    word_doc = word_loader.load()
                    text += word_doc[0].page_content

        st.success('Document loaded!', icon="✅")
    st.write(f'Loaded document:\n------------\n\n{text}')
    
    with st.sidebar:
        with st.spinner('Splitting the document into chunks...'):
            #document chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1028,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_text(text)
            document_chunks = text_splitter.create_documents(chunks)
        st.success(f'Document chunking completed! {len(chunks)} chunks', icon="✅")
    st.write(f'all {len(document_chunks)} chunks are following -\n\n')
    for i in range(len(document_chunks)):
         st.write(f'chunk number {i}:\n------------\n\n{document_chunks[i]}')

    with st.sidebar:
        with st.spinner('Creating vectorstore from the document chunks...'):
            embeddings = OllamaEmbeddings(model="nomic-embed-text",show_progress=True)
            persist_directory = "./local_embeddings/embeddings5"
            vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory=persist_directory)
            #load the Chroma database from disk
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            
        st.success('Embeddings created and saved to vectorstore', icon="✅")
        st.info("This vector store will take care of storing embedded data and perform vector search for you.")
    
    return vector_store

def get_conversational_retrieval_chain(history_aware_retriever):

    prompt_get_answer = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based only on the following context:\n\n{context} and Do not make up stuff you don't know."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    llm = ChatOllama(model="mistral:latest", temperature=0)
    document_chain = create_stuff_documents_chain(llm, prompt_get_answer)
    return create_retrieval_chain(history_aware_retriever, document_chain)

def get_history_aware_retriever(vector_store):
     
    prompt_search_query = ChatPromptTemplate.from_messages([
         MessagesPlaceholder(variable_name='chat_history'),
         "user", "{input}",
         "user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Do not make up stuff you don't know."
    ])
    
    llm = ChatOllama(model="mistral:latest", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    history_aware_retriever = create_history_aware_retriever(
         llm=llm,
         retriever=retriever,
         prompt=prompt_search_query
    )

    return history_aware_retriever


def get_response(user_input):
    history_aware_retriever = get_history_aware_retriever(st.session_state.vector_store)
    conversational_retrieval_chain = get_conversational_retrieval_chain(history_aware_retriever) #to actually answer the user question
    response = conversational_retrieval_chain.invoke({
         "chat_history": st.session_state.chat_history,
         "input": user_input
    })
    with st.sidebar:
         st.write(response)
    return response['answer']


# app config
st.set_page_config(page_title="Fully Local RAG App", page_icon="")
st.title("Fully Local RAG App")

with st.sidebar:
    st.subheader("Your documents")
    uploaded_docs = st.file_uploader(
        "Upload your files here and click on Process. Allowed extensions: .pdf, .docx", 
        type=(["pdf",".docx"]), 
        accept_multiple_files=True)
    process_button = st.button("Process")

if uploaded_docs == []:
    st.info("Please upload your files, then click on Process")
    print("uploaded_docs currently is empty: ", uploaded_docs)

else:
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am your Fully Local RAG App. How can I help you?")
            ]   
    if "conversation" not in st.session_state:
            st.session_state.conversation = None

    if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
    
    #track if st.button("Process") is clicked
    if "button_clicked" not in st.session_state:
         st.session_state.button_clicked = 0

    if process_button: 
        st.session_state.button_clicked = 1
        print("button clicked!")
        #build the vectorstore from uploaded_docs
        st.session_state.vector_store = get_vectorstore_from_pdf(uploaded_docs)

    if st.session_state.button_clicked == 1:
        #user input
        user_query = st.chat_input("Type your message here...")
        print(f"user_query: {user_query}")
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            response = get_response(user_query)
            st.session_state.chat_history.append(AIMessage(content=response))
            with st.sidebar:
                 st.subheader("st.session_state.chat_history")
                 st.write(st.session_state.chat_history)

        # show the HumanMessage and AIMessage as conversation on the webpage
        for message in st.session_state.chat_history:
            # st.write(st.session_state.chat_history)
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
