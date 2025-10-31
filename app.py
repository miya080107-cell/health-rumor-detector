# app.py
from flask import Flask, request, jsonify, send_from_directory
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from flask_cors import CORS

# --- 加载环境变量 ---
load_dotenv()

# 从 .env 中读取密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("请在 .env 中配置 DEEPSEEK_API_KEY")

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# --- 配置 ---
MODEL_NAME = "deepseek-chat"   # 或 deepseek-reasoner
LOGS_CSV = "logs.csv"


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://health-rumor-detector.onrender.com"}})

def build_prompt(user_text: str) -> str:
    prompt = f"""
You are a careful medical-information fact-checking assistant.

A user will provide a short statement about a **disease, symptom, treatment, or health claim**.

Your task:
1) Judge whether the user's statement is **accurate**, **partially correct**, or **medical misinformation (rumor)**.
2) Provide a short, clear explanation (1–3 sentences) to clarify the truth.
3) Provide **authoritative reference links ONLY** from scientific papers or recognized medical organizations (e.g., WHO, NIH, CDC, PubMed, Mayo Clinic). 
4) Reply in **STRICT JSON format** with keys: conclusion, explanation, sources (list).

User statement:
\"\"\"{user_text}\"\"\"

Respond strictly in JSON like:
{{
  "conclusion": "accurate",
  "explanation": "Short reason with clarification.",
  "sources": [
     {{"title": "Relevant Scientific Source", "link": "https://www.ncbi.nlm.nih.gov/pmc/articles/"}}
  ]
}}

IMPORTANT:
- Output must be valid JSON only (no markdown or extra text).
- Keys and values must use double quotes.
- Do NOT include comments or trailing commas.
- Do NOT wrap the JSON in code blocks.
"""
    return prompt


def call_model(prompt: str, retries=2):
    """
    调用 DeepSeek AI，并保证返回严格 JSON。
    如果模型输出为空或不可解析，会尝试重试。
    """
    import time
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )

            # DeepSeek 最新接口返回对象属性为 .message.content
            try:
                text = response.choices[0].message.content.strip()
            except AttributeError:
                text = ""

            # 尝试解析 JSON
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_text = text[start:end]
                parsed = json.loads(json_text)
            except Exception:
                parsed = None

            # 如果解析成功，返回
            if parsed:
                for key in ["conclusion", "explanation", "sources"]:
                    if key not in parsed:
                        parsed[key] = [] if key == "sources" else "unknown"
                return json.dumps(parsed, ensure_ascii=False), response

            # 如果失败，等待 1 秒重试
            time.sleep(1)

        except Exception as e:
            last_error = str(e)
            time.sleep(1)

    # 所有尝试失败，返回默认结构
    return json.dumps({
        "conclusion": "unknown",
        "explanation": f"DeepSeek API 调用失败或模型返回不可解析: {last_error if 'last_error' in locals() else ''}",
        "sources": [{"title": "Example Scientific Source", "link": "https://www.ncbi.nlm.nih.gov/pmc/articles/"}]
    }, ensure_ascii=False), None

import csv

def append_log(entry: dict):
    file_exists = os.path.exists(LOGS_CSV)
    with open(LOGS_CSV, mode='a', newline='', encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "user_text", "result"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

# 支持前端 index.html
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    user_text = data.get("text", "").strip()
    if not user_text:
        return jsonify({"error": "No text provided."}), 400

    prompt = build_prompt(user_text)

    try:
        model_text, raw_response = call_model(prompt)
    except Exception as e:
        return jsonify({"error": f"DeepSeek API failed: {str(e)}"}), 500

    # 尝试解析 JSON
    try:
        start = model_text.find("{")
        end = model_text.rfind("}") + 1
        json_text = model_text[start:end]
        parsed = json.loads(json_text)
    except Exception:
        parsed = {
            "conclusion": "unknown",
            "explanation": "模型返回不可解析的结果。",
            "sources": [{"title": "Example Scientific Source", "link": "https://www.ncbi.nlm.nih.gov/pmc/articles/"}]
        }

    # 保证 sources 至少有占位科研文章
    if parsed.get("sources") in (None, [], ""):
        parsed["sources"] = [{"title": "Example Scientific Source", "link": "https://www.ncbi.nlm.nih.gov/pmc/articles/"}]

    result = {
        "conclusion": parsed.get("conclusion"),
        "explanation": parsed.get("explanation"),
        "sources": parsed.get("sources"),
        "raw_model_output": model_text
    }

    # 记录日志
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_text": user_text,
        "result": json.dumps(result, ensure_ascii=False)
    }
    append_log(log_entry)

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
