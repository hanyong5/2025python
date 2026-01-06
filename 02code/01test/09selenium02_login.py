from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.common.action_chains import ActionChains
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import time,random

# pip install pyperclip ctrl+c, ctrl+v
import pyperclip

import os
from dotenv import load_dotenv

load_dotenv(override=True)

LOGIN_ID = os.getenv('LOGIN_ID')
LOGIN_PW = os.getenv('LOGIN_PW')

url = "https://nid.naver.com/nidlogin.login"

# selenium 설정
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
driver = webdriver.Chrome(options=options)

# options.add_argument('--start-maximized')
# options.add_argument('--disable-blink-features=AutomationControlled')
# options.add_argument('--no-sandbox')
# options.add_argument('--disable-dev-shm-usage')
# options.add_argument('--disable-gpu')

# options.add_argument(
#     "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#     "AppleWebKit/537.36 (KHTML, like Gecko) "
#     "Chrome/120.0.0.0 Safari/537.36"
# )

# options.add_experimental_option("excludeSwitches", ["enable-automation"])
# options.add_experimental_option("useAutomationExtension", False)

# driver = webdriver.Chrome(options=options)

# driver.execute_cdp_cmd(
#     "Page.addScriptToEvaluateOnNewDocument",
#     {
#         "source": """
#         Object.defineProperty(navigator, 'webdriver', {
#             get: () => undefined
#         })
#         """
#     }
# )



driver.get(url)

# id 입력
id_input = driver.find_element(By.ID,"id")
pyperclip.copy(LOGIN_ID) # CTRL+C
id_input.click()
id_input.send_keys(Keys.CONTROL,'v') # ctrl+v
time.sleep(random.uniform(1.5,3.5))

# pw 입력
pw_input = driver.find_element(By.ID,"pw")
pyperclip.copy(LOGIN_PW) # CTRL+C
pw_input.click()
pw_input.send_keys(Keys.CONTROL,'v') # ctrl+v
time.sleep(random.uniform(1.5,3.5))

# button 클릭
login_btn = driver.find_element(By.ID,"log.login")
login_btn.click()
time.sleep(random.uniform(1.5,3.5))

# 글작성페이지 이동
driver.get("https://blog.naver.com/GoBlogWrite.naver")
time.sleep(random.uniform(1.5,3.5))


# 타이핑 속도 설정
TYPE_DELAY = 0.01  # 초 단위

# 입력할 제목과 내용
title_text = "안녕하세요, 여행을 사랑하는 모든 분들께!"
body_text = (
    "설레는 발걸음으로 시작되는 여행의 순간들, 낯선 거리에서 마주치는 예상치 못한 만남들, "
    "그리고 새로운 문화와 풍경 속에서 느끼는 감동을 이 공간에 담아보려 합니다. "
    "여행은 단순한 이동이 아닌, 나를 발견하고 세상을 더 넓게 바라보는 소중한 시간이라고 생각합니다.\n"
    "앞으로 제가 경험한 특별한 순간들과 여행 팁을 여러분과 나누며, 함께 성장하는 여행 커뮤니티를 만들어가길 희망합니다. "
    "여러분의 발자국도 이 여정에 함께해주세요!"
)


# 1. iframe 전환
driver.switch_to.frame(driver.find_element(By.ID, "mainFrame"))
time.sleep(2)

# 2. 팝업 닫기 (.se-popup-button-cancel)
try:
    cancel_btn = driver.find_element(By.CLASS_NAME, "se-popup-button-cancel")
    cancel_btn.click()
    print("팝업 취소 버튼 클릭됨")
    time.sleep(1)
except:
    print("팝업 없음 (se-popup-button-cancel)")

# 3. 도움말 닫기 (.se-help-panel-close-button)
try:
    help_close_btn = driver.find_element(By.CLASS_NAME, "se-help-panel-close-button")
    help_close_btn.click()
    print("도움말 닫기 버튼 클릭됨")
    time.sleep(1)
except:
    print("도움말 없음 (se-help-panel-close-button)")

# 4. 제목 입력
title_input = driver.find_element(By.CSS_SELECTOR, ".se-section-documentTitle")
title_input.click()
actions = ActionChains(driver)
for char in title_text:
    actions.send_keys(char).perform()
    time.sleep(TYPE_DELAY)

# # 5. 본문 입력
content_area = driver.find_element(By.CSS_SELECTOR, ".se-section-text")
content_area.click()
actions = ActionChains(driver)
for char in body_text:
    actions.send_keys('\n' if char == '\n' else char).perform()
    time.sleep(TYPE_DELAY)


    

# 6. 저장 버튼 클릭
save_button = driver.find_element(By.CLASS_NAME, "save_btn__bzc5B")
save_button.click()
print("✅ 글 저장 완료!")

# 유지 또는 종료
# input("엔터를 누르면 종료됩니다...")
# driver.quit()




try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("프로그램 종료")
    driver.quit()




# time.sleep(10)
# driver.quit()