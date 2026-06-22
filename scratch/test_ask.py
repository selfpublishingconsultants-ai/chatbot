import requests

url = "http://127.0.0.1:5000/ask"
payload = {
    "question": "i wanna talk to human",
    "session_id": "test_session_123"
}

try:
    response = requests.post(url, json=payload)
    print("STATUS CODE:", response.status_code)
    print("RESPONSE JSON:", response.json())
except Exception as e:
    print("ERROR:", e)
