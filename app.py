# app.py
import os
import json
import redis
import tempfile
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from typing import Dict, Any

# --- Redis 多用户会话记忆（24 小时过期） ---
REDIS = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
def load_history(user_id: str):
    data = REDIS.get(f"conv:{user_id}")
    return json.loads(data) if data else []
def save_history(user_id: str, history, ttl_hours=24):
    REDIS.setex(f"conv:{user_id}", ttl_hours*3600, json.dumps(history, ensure_ascii=False))
print(f"[DEBUG] Saving history: {history}")
print(f"[DEBUG] Loaded history: {history}")
# --- 导入你原封不动的所有函数 ---
import orig_funcs

# --- OpenAI Client & 函数注册 ---
openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

function_specs = [
    {
        "name": "math",
        "description": "數學運算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "time",
        "description": "獲取當前時間/日期",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "weather",
        "description": "天氣查詢，包括過去、現在和未來的天氣。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search",
        "description": "網頁搜索",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "doc_qa",
        "description": "單文檔 QA",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "file": {"type": "object"}
            },
            "required": ["query", "file"]
        }
    },
    {
        "name": "doc_summarisation",
        "description": "單文檔總結",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "file": {"type": "object"}
            },
            "required": ["query", "file"]
        }
    },
    {
        "name": "multi_doc_qa",
        "description": "多文檔 QA / Summarize",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "data": {"type": "string", "format": "binary"}
                        }
                    }
                }
            },
            "required": ["query"]
        }
    }
]

# map name->orig_funcs 中的对应函数
fn_map = {
    "math": lambda args: orig_funcs._calc_tool(args["expression"]),
    "time": lambda args: orig_funcs.get_time_tool(args["query"]),
    "weather": lambda args: orig_funcs.weather_agent_tool(args["query"]),
    "search": lambda args: orig_funcs.search_web(args["query"]),
    "doc_qa": lambda args: orig_funcs.uploaded_qa(args),
    "doc_summarisation": lambda args: orig_funcs.document_summarize(args),
    "multi_doc_qa": lambda args: orig_funcs.autogen_multi_document_analysis(
    args["query"],
    [f["data"] for f in args["files"]],
    [f["name"] for f in args["files"]]
)
}

# --- FastAPI 服务 ---
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    body    = await request.json()
    user_id = body.get("user_id", "anonymous")
    #msg_raw = body.get("query", "")
    msg_raw = body.get("query", "\"\"")
    try:
        msg = json.loads(msg_raw)
    except json.JSONDecodeError:
        msg = msg_raw  # fallback: 如果解不開就用原始字串
    files   = body.get("files", [])  # [{"name":..,"data":..}, ...]

    # 读历史
    history = load_history(user_id)
    if not history:
        history.append({"role":"system","content":"你是一名 AI 助理，支持數學、時間、天氣、搜索、文檔QA和文檔總結。"})
    history.append({"role":"user","content":msg})
    
    fallback_keywords = ["天氣", "weather"]
    if any(keyword in msg for keyword in fallback_keywords):
        print("[DEBUG] Triggered fallback: forcing weather function")
        result = fn_map["weather"]({"query": msg})
        print(f"[DEBUG] Fallback triggered, weather result: {result}")
        
        # 再次向 GPT 匯總結果
        follow = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                *history,
                {"role": "function", "name": "weather", "content": json.dumps(result, ensure_ascii=False)},
                {"role": "user", "content": "請根據工具返回的數據，用簡明的自然語言回答用戶。"}
            ]
        )
        reply = follow.choices[0].message.content

        history.append({"role": "assistant", "content": reply})
        save_history(user_id, history)
        return JSONResponse({"reply": reply})
        
    # 1) 询问 GPT 是否调用函数
    gpt_resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        functions=function_specs,
        function_call="auto"
    )
    msg_obj = gpt_resp.choices[0].message

    if msg_obj.function_call:
        name = msg_obj.function_call.name
        args = json.loads(msg_obj.function_call.arguments or "{}")
        # 携带上传文件
        if name == "doc_qa" and files:
            args["file"] = files[0]
        if name == "multi_doc_qa":
            args["files"] = files

        # 调用原函数
        result = fn_map[name](args)

        # 2) 再次向 GPT 汇总结果
        follow = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                *history,
                {"role":"function","name":name,"content":json.dumps(result,ensure_ascii=False)},
                {"role":"user","content":"請根據工具返回的數據，用簡明的自然語言回答用戶。"}
            ]
        )
        reply = follow.choices[0].message.content

    else:
        # 直接回答
        reply = msg_obj.content

    # 存历史 & 返回
    history.append({"role":"assistant","content":reply})
    save_history(user_id, history)
    return JSONResponse({"reply": reply})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
