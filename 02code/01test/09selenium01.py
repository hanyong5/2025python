from selenium import webdriver
from selenium.webdriver.common.by import By
import time

import os
from dotenv import load_dotenv

load_dotenv()
driver = webdriver.Chrome()

driver.get("https://example.com")

title = driver.find_element(By.TAG_NAME,'h1').text
ptag = driver.find_element(By.TAG_NAME,'p').text
print(f'h1 tag text : {title}')
print(f'p tag text : {ptag}')

secret_key = os.getenv('SECRET_KEY')

print(secret_key)

time.sleep(10)
driver.quit()
