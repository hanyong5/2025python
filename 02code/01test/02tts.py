from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame
import time


text = "박한별, 남편 논란 딛고 6년"

# 1.번역
translated = GoogleTranslator(source="auto",target="en").translate(text)
# print(result)

# 2. tts
tts = gTTS(text=translated,lang="en")
tts.save("voice.mp3")

# 3. 재생
pygame.mixer.init()
pygame.mixer.music.load('voice.mp3')
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    time.sleep(0.1)
