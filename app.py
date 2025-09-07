# # import these dependencies: pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai hugginface_hub
# # for performing embeddings for free using Instructor (which is better than openAIEmbeddings according to hugginface rankings) : pip install InstructorEmbedding sentence_transformers

# # starting with Graphical user interface
# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader # pyright: ignore[reportMissingImports]
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# # from langchain.llms import openai
# from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size = 1000,
#         chunk_overlap = 200,
#         length_function = len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings() 
#     #embeddings = HuggingFaceInstructEmbeddings(model_name ="hkunlp/instructor-xl") #, model_kwargs={"device": "cpu"})   # or "cuda" if you have GPU) 
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) # to understand how the memory works in langchain, and buffermemory, check out the video he mentioned at timestampt 43:30.
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm = llm,
#         retriever = vectorstore.as_retriever(),
#         memory = memory
#     )
#     return conversation_chain

# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     # st.write(response) # this gives the entire object with everything asked
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.chat_history):
#         if i % 2 == 0:
#             st.write("👤 You:", message.content)
#         else:
#             st.write("🤖 Bot:", message.content)

# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#         if "chat_history" not in st.session_state:
#             st.session_state.chat_history = None


#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")      # check out 55 for handling user query
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDF's here and press on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)
#                 #st.write(raw_text)

#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)
#                 # st.write(text_chunks)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create coversation chain
#                 st.session_state.conversation = get_conversation_chain(vectorstore) # 'st.session_state.' listen to the explaination @ 46:30, very important note about streamlit and context retrieval


# # this is make sure that this app (code in the main function is being used only here and not pulled anwhere else)
# if __name__ == '__main__':
#     main()


import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF's here and press on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
