from selenium import webdriver
from selenium.webdriver.common.by import By
import time

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
driver.get(url)

id_input = driver.find_element(By.ID,"id")


time.sleep(5)
driver.quit()