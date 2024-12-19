import os
import textwrap

import chromadb
import langchain
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, YoutubeLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from pdf2image import convert_from_path
import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


# Load mô hình
from getpass import getpass
OPENAI_API_KEY = getpass("Nhập pass: ")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
model = OpenAI(temperature=1, model_name="gpt-3.5-turbo-0125")

# Chuyển file PDF về dạng text
# pdf_loader = UnstructuredPDFLoader("Li_thuyet_Hadoop.pdf")
# pdf_pages = pdf_loader.load_and_split()


def process_multiple_pdfs(pdf_directory):
    all_texts = []  # Danh sách để lưu tất cả văn bản từ các file PDF

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):  # Chỉ xử lý file có định dạng PDF
            file_path = os.path.join(pdf_directory, filename)
            print(f"Đang xử lý file: {file_path}")

            # Load và chuyển PDF thành văn bản
            pdf_loader = UnstructuredPDFLoader(file_path)
            pdf_pages = pdf_loader.load_and_split()

            # Chia nhỏ văn bản
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
            texts = text_splitter.split_documents(pdf_pages)

            # Thêm vào danh sách tất cả văn bản
            all_texts.extend(texts)

    return all_texts

# Text Splitters
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
#texts = text_splitter.split_documents(pdf_pages)
#len(texts)

# Chuyển đổi toàn bộ file PDF trong thư mục
texts = process_multiple_pdfs('filePdf')
print(f"Đã xử lý {len(texts)} đoạn văn bản từ các file PDF.")

# Sử dụng mô hình embedding
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Chuyển toàn bộ text thông qua mô hình embedding về dạng vector và lưu dưới dạng db
db = Chroma.from_documents(texts, hf_embeddings, persist_directory="db")


# Prompt đơn giản được sử dụng
custom_prompt_template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt

Context: {context}
Question: {question}

"""

# Hàm khởi tạo Prompt sẽ sử dụng
from langchain import PromptTemplate
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt



# Khai báo prompt và chain
prompt = set_custom_prompt()
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={'prompt': prompt}
)

def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=100)))

def ask_questions():
    print("Bạn có thể đặt câu hỏi về tài liệu. Gõ 'exit' để thoát.")
    while True:
        query = input("\nCâu hỏi của bạn: ")
        if query.lower() == "exit":
            print("Kết thúc chương trình. Tạm biệt!")
            break
        try:
            response = chain.run(query)
            print_response(response)
        except Exception as e:
            print("Đã xảy ra lỗi:", e)

# Gọi hàm hỏi đáp
ask_questions()


