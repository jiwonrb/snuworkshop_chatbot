import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage 
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from dotenv import load_dotenv

load_dotenv()

# handle streaming conversation
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
    
    def on_llm_end(self, **kwargs) -> None:
        self.text += "\n\n마음에 드냐옹? 💕 언제든 추가로 질문하라냥! 🐾"
        self.container.markdown(self.text)

# function to extract text from an HWP file
import olefile
import zlib
import struct

def get_hwp_text(filename):
    f = olefile.OleFileIO(filename)
    dirs = f.listdir()

    # HWP 파일 검증
    if ["FileHeader"] not in dirs or \
       ["\x05HwpSummaryInformation"] not in dirs:
        raise Exception("Not Valid HWP.")

    # 문서 포맷 압축 여부 확인
    header = f.openstream("FileHeader")
    header_data = header.read()
    is_compressed = (header_data[36] & 1) == 1

    # Body Sections 불러오기
    nums = []
    for d in dirs:
        if d[0] == "BodyText":
            nums.append(int(d[1][len("Section"):]))
    sections = ["BodyText/Section"+str(x) for x in sorted(nums)]

    # 전체 text 추출
    text = ""
    for section in sections:
        bodytext = f.openstream(section)
        data = bodytext.read()
        if is_compressed:
            unpacked_data = zlib.decompress(data, -15)
        else:
            unpacked_data = data
    
        # 각 Section 내 text 추출    
        section_text = ""
        i = 0
        size = len(unpacked_data)
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3ff
            rec_len = (header >> 20) & 0xfff

            if rec_type in [67]:
                rec_data = unpacked_data[i+4:i+4+rec_len]
                section_text += rec_data.decode('utf-16')
                section_text += "\n"

            i += 4 + rec_len

        text += section_text
        text += "\n"

    return text

# Function to extract text from an PDF file
from pdfminer.high_level import extract_text

def get_pdf_text(filename):
    raw_text = extract_text(filename)
    return raw_text

# document preprocess
def process_uploaded_file(uploaded_file):
    # Load document if file is uploaded
    if uploaded_file is not None:
        
        # loader
        if uploaded_file.type == 'application/pdf':
            raw_text = get_pdf_text(uploaded_file)
        elif uploaded_file.type == 'application/octet-stream':
            raw_text = get_hwp_text(uploaded_file)
            
        # splitter
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,
        )
        all_splits = text_splitter.create_documents([raw_text])

        print("총 " + str(len(all_splits)) + "개의 passage")

        # storage
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
                
        return vectorstore, raw_text
    return None

# generate response using RAG technic
def generate_response(query_text, vectorstore, callback):

    # retriever 
    docs_list = vectorstore.similarity_search(query_text, k=3)
    docs = ""
    for i, doc in enumerate(docs_list):
        docs += f"'문서{i+1}':{doc.page_content}\n"
        
    # generator
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[callback])
    
    # chaining
    rag_prompt = [
        SystemMessage(
            content="너는 문서에 대해 질의응답을 하는 고양이야. 주어진 자료를 참고하여 사용자의 질문에 답변을 해줘. 문서에 내용이 정확하게 나와있지 않으면 웹 검색을 통해 알려줘. 모든 문장의 끝을 귀엽게 'meow' 또는 '냥'으로 마무리해줘!"
        ),
        HumanMessage(
            content=f"질문:{query_text}\n\n{docs}"
        ),
    ]

    response = llm(rag_prompt)
    
    return response.content

def generate_summarize(raw_text, callback, language):

    # generator 
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[callback])
    
    if language == 'ko':
        system_message = "다음 나올 문서를 'Notion style'로 적절한 이모지를 불렛포인트로 사용해서 요약해줘. 중요한 내용만. 모든 문장의 끝에 '냥'을 붙여줘. 또한 '~다냥'과 같은 자연스러운 문장으로 끝나게 해줘."
    else:
        system_message = "Summarize the following document in 'Notion style' using appropriate emojis as bullet points. Focus on the important content only and end each sentence with 'meow'."

    # prompt formatting
    rag_prompt = [
        SystemMessage(
            content=system_message
        ),
        HumanMessage(
            content=raw_text
        ),
    ]
    
    response = llm(rag_prompt)
    return response.content

# page title
st.set_page_config(page_title=' 🧊 꽁꽁 얼어붙은 논문 위로 🐈 고양이가 걸어다닙니다 🐾')
st.title('🧊 꽁꽁 얼어붙은 논문 위로 🐈 고양이가 걸어다닙니다 🐾')

# enter token
import os
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
save_button = st.sidebar.button("Save Key")
if save_button and len(api_key)>10:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API Key saved successfully!")

# file upload
uploaded_file = st.file_uploader('Upload an document', type=['hwp','pdf'])

# file upload logic
if uploaded_file:
    vectorstore, raw_text = process_uploaded_file(uploaded_file)
    if vectorstore:
        st.session_state['vectorstore'] = vectorstore
        st.session_state['raw_text'] = raw_text
        
# chatbot greetings - 첫 인사
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content="안냥! 바쁜 고양이들을 위해 논문을 업로드하면 귀엽게 요약해주겠다냥! 🐾"
        )
    ]

# conversation history print 
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)
    
# message interaction
if prompt := st.chat_input("영문 요약은 'sum', 한글 요약은 '요약'이라고 입력하라냥🐈"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        
        if prompt == "요약":
            response = generate_summarize(st.session_state['raw_text'], stream_handler, language='ko')
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response + "\n\n마음에 드냐옹? 💕 언제든 추가로 질문하라냥! 🐾")
            )

        elif prompt == "sum":
            response = generate_summarize(st.session_state['raw_text'], stream_handler, language='en')
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response + "\n\nDo you like it? 💕 Feel free to ask more questions, meow! 🐾")
            )
        else:
            response = generate_response(prompt, st.session_state['vectorstore'], stream_handler)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response + "\n\n마음에 드냐옹? 💕 언제든 추가로 질문하라냥! 🐾")
            )
