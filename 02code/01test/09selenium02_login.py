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
# options.add_argument('--start-maximized')
# driver = webdriver.Chrome(options=options)


options.add_argument('--start-maximized')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')

options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

driver = webdriver.Chrome(options=options)

driver.execute_cdp_cmd(
    "Page.addScriptToEvaluateOnNewDocument",
    {
        "source": """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        })
        """
    }
)



driver.get(url)

id_input = driver.find_element(By.ID,"id")


time.sleep(5)
driver.quit()