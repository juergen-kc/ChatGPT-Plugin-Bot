import requests
import json

def test_chat():
    url = 'http://localhost:5003/ask'
    question = {
        'query': 'What is the most popular plugin?'
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, headers=headers, data=json.dumps(question))

    if response.status_code == 200:
        print("Response from server: ", response.json())
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    test_chat()
