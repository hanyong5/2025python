from PyPDF2 import PdfReader
import tiktoken


pdf_path = "data/sample01.pdf"



def extract_text_from_pdf(pdf_path:str)->str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for i,page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text: 
                text += page_text + "\n"

    return text


pdf_text = extract_text_from_pdf(pdf_path)
enc = tiktoken.encoding_for_model("gpt-4o-mini")


print("글자 수:", len(pdf_text))
print("토큰 수:", len(enc.encode(pdf_text)))


