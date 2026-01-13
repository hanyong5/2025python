from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv(override=True)

# print(os.getenv("OPENAI_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",
    # input="서울에 갈 만한 여행지 추천해줘"
    input=[
        {
            "role":"system",
            "content":(
                "너는 json 생성기다"
                "반드시 json형식로만 출력해줘"
                "설명, 말머리, 마크다운, 코드블록은 절대 포함하지 마"
                "제목, 설명내용, 5개만 json출력"
            )
        },
        {
            "role":"user",
            "content":"서울에 갈 만한 여행지 추천해줘"
        }
    ]
)

print(response.output_text)