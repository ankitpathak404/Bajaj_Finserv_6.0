import requests
from bs4 import BeautifulSoup
import asyncio

def extract_token_from_url(url):

    url="https://register.hackrx.in/utils/get-secret-token?hackTeam=5100"
    response = requests.get(url)
    print(url,"\n")
    response.raise_for_status()  # ensure HTTP 200
    soup = BeautifulSoup(response.text, 'html.parser')
    token_div = soup.find('div', id='token')
    return token_div.text.strip() if token_div else None

url = "https://register.hackrx.in/utils/get-secret-token?hackTeam=5100"
token = extract_token_from_url(url)
print("Token:", token)
