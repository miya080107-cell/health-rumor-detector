# test_api.py
import requests
import json

# 你的本地 Flask API 地址
URL = "http://127.0.0.1:5000"

# 测试语句，可以改成任意 PCOS 谣言或说法
test_text = "吃大量糖会直接导致 PCOS。"

payload = {
    "text": test_text
}

try:
    response = requests.post(URL, json=payload)
    response.raise_for_status()  # 如果状态码不是 200，会抛出异常
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
