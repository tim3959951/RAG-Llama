# app.py
import os
import json
import redis
import tempfile
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from typing import Dict, Any
from langsmith import traceable
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

# --- Redis 多用户会话记忆（使用 BufferMemory） ---
REDIS = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)

# 獲取用戶的記憶（使用 BufferMemory 管理）
def get_memory(user_id: str, k=10):
    """獲取用戶的 BufferMemory，限制保留最近 k 條消息"""
    memory = ConversationBufferWindowMemory(k=k, return_messages=True)
    
    # 從 Redis 讀取序列化的記憶
    data = REDIS.get(f"memory:{user_id}")
    if data:
        try:
            stored_messages = json.loads(data)
            for msg in stored_messages:
                if msg["type"] == "human":
                    memory.chat_memory.add_user_message(msg["content"])
                elif msg["type"] == "ai":
                    memory.chat_memory.add_ai_message(msg["content"])
        except Exception as e:
            print(f"[DEBUG] Error loading memory: {e}")
    
    return memory

# 保存用戶的記憶
def save_memory(user_id: str, memory, ttl_hours=24):
    """將用戶的 BufferMemory 保存到 Redis"""
    messages = []
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"type": "ai", "content": msg.content})
    
    REDIS.setex(f"memory:{user_id}", ttl_hours*3600, json.dumps(messages, ensure_ascii=False))

# 將 BufferMemory 轉換為 OpenAI 格式的歷史記錄
def memory_to_history(memory):
    """將 BufferMemory 轉換為可用於 OpenAI API 的消息格式"""
    history = [{"role": "system", "content": "你是一名 AI 助理，支持數學、時間、天氣、搜索、文檔QA和文檔總結。"}]
    
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    
    return history

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
    msg_raw = body.get("query", "\"\"")
    try:
        msg = json.loads(msg_raw)
    except json.JSONDecodeError:
        msg = msg_raw  # fallback: 如果解不開就用原始字串
    print(f"[DEBUG] User Query: {msg}")
    files   = body.get("files", [])  # [{"name":..,"data":..}, ...]

    # 獲取記憶並將用戶消息添加到記憶
    memory = get_memory(user_id)
    memory.chat_memory.add_user_message(msg)
    
    # 將記憶轉換為 OpenAI 格式的歷史記錄
    history = memory_to_history(memory)
    
    fallback_keywords = ["weather", "rain", "snow", "cold", "hot", "sunscreen", "sunglasses", "umbrella", "windy", "cloudy", "sunny", "temperature", "forecast", "天氣", "下雨", "冷嗎", "熱嗎", "氣溫"]
    msg_str = msg if isinstance(msg, str) else str(msg)
    if any(keyword in msg_str for keyword in fallback_keywords):
        print("[DEBUG] Triggered fallback: forcing weather function")
        result = fn_map["weather"]({"query": msg_str})

        # 再次向 GPT 匯總結果（注意這裡去掉 json.dumps）
        follow = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                *history,
                {"role": "function", "name": "weather", "content": result},
                {"role": "user", "content": f"用戶的原始提問是：「{msg_str}」。請完全基於工具提供的數據，用與用戶相同的語言簡明回答(若無法判斷則用繁體中文)。不要使用模型的自有知識，也不要回應「沒有資料」「查不到」「無法提供」這類話，只需要根據工具的結果生成對應語言的自然語言回答。"}
            ]
        )
        reply = follow.choices[0].message.content
        print(f"[DEBUG] Final Output to User (Fallback): {reply}")
        
        # 將回覆添加到記憶並保存
        memory.chat_memory.add_ai_message(reply)
        save_memory(user_id, memory)
        
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
        print(f"[DEBUG] Final Output to User (function call): {reply}")
    else:
        # 直接回答
        reply = msg_obj.content
        print(f"[DEBUG] Final Output to User (no function call): {reply}")
    
    # 將回覆添加到記憶並保存
    memory.chat_memory.add_ai_message(reply)
    save_memory(user_id, memory)
    
    return JSONResponse({"reply": reply})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
