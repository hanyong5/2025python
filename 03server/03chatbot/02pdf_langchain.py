from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import tiktoken

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


def extract_text_from_pdf(pdf_path:str)->str:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = "\n".join(doc.page_content for doc in documents)
    return text

def split_text_into_chunks(text:str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def create_vactor_store(chunks:list):
    embaddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    return FAISS.from_texts(chunks,embaddings)


def answer_question(question:str,knowledge_base)->str:
    docs = knowledge_base.similarity_search(question, k=5)

    print("\n [검색된 관련 청크들]")
    for i,doc in enumerate(docs,1):
        preview = doc.page_content[:150].replace('\n',' ')
        print(f'{i}. {preview}...')



def main():
    pdf_path = "data/summary.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)
    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    print(f'글자 수 : {len(pdf_text)}')
    print(f'토큰 수 : {len(enc.encode(pdf_text))}')

    # chunk로 나누기
    chunks = split_text_into_chunks(pdf_text)
    print(f'총 chunk 수 : {len(chunks)}')

    #각 chunk 정보 출력
    # for i,chunk in enumerate(chunks,1):
    #     token_count = len(enc.encode(chunk))
    #     print(f'chunk {i} : 글자수 {len(chunk)}, 토큰 수 {token_count}')

    print("\n----------백터스토어생성---------")
    knowledge_base= create_vactor_store(chunks)

    print("\n----------질문응답테스트---------")
    question = "where can i use chatGPT?"
    print(f'질문 : {question}')


    answer = answer_question(question, knowledge_base)
    print(f' \n  답변: \n {answer}')


if __name__ == "__main__":
    main()