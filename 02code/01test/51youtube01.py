from youtube_transcript_api import YouTubeTranscriptApi
import re

# text = "hello world"
# match = re.search(r"world",text)
# if match:
#     print("찾았다")
# https://www.youtube.com/watch?v=2cmwG0EDwKo 


def extract_video_id(url):
    match = re.search(r"(?:v=)([^&]+)",url)
    return  match.group(1) if match else None

def get_transcript(video_id):
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id,languages=["ko","en"]).to_raw_data()
    full_text = "\n".join([item['text'] for item in transcript])
    return full_text


if __name__ == "__main__":
    # print("자동실행합니다.")
    youtube_url = input("경로를 입력하세요: ").strip()
    video_id = extract_video_id(youtube_url)
    print(f'값추출 : {video_id}')

    if video_id:
        text = get_transcript(video_id)
        print(text)
    else:
        print("올바른 url을 입력하세요")
