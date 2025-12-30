# 1. python 이미지
FROM python:3.10-slim

WORKDIR /app

RUN pip install streamlit deep-translator gTTS

COPY . .

EXPOSE 8501


# streamlit run main.py
CMD ["streamlit","run","main.py","--server.address=0.0.0.0"]