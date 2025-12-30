import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS

st.title("번역기")

# 한글입력
ktext = st.text_input(
    label="번역할 문장",
    placeholder="한글로 입력하세요"
)

lang = st.selectbox(
    "번역할 언어선택",
    ["영어", "일어", "한국어"]
)

lang_map = {
    "영어":"en",
    "일어":"ja",
    "한국어":"ko"
}

# 버튼생성
if st.button("번역하기"):
    if ktext.strip() == "":
        st.warning("한글문장을 입력해주세요")
    else:
        translated = GoogleTranslator(
            source="auto",
            target=lang_map[lang]
        ).translate(ktext)

        st.success("번역결과")
        st.write(translated)

        tts = gTTS(text=translated,lang=lang_map[lang])
        tts.save("voice.mp3")

        st.audio("voice.mp3")

        
        