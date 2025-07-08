import threading
import requests
from bs4 import BeautifulSoup

urls = [
    'https://python.langchain.com/docs/concepts/',
    'https://python.langchain.com/docs/introduction/',
    'https://python.langchain.com/docs/tutorials/'
]

def fetch_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        first_tag = soup.body
        if first_tag:
            next_element = first_tag.next
            print(f"Next element after <body> in {url} is: {repr(next_element)}")
        else:
            print(f"No <body> tag found in {url}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")

threads = []

for url in urls:
    thread = threading.Thread(target=fetch_content, args=(url,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
