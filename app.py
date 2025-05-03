import os
import json
import tempfile
import shutil
import torch
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from openai import OpenAI
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# === Environment setup ===
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/tmp/huggingface/metrics"
os.environ["AUTOGEN_WORKSPACE"] = "/tmp/autogen_workspace"

os.makedirs(os.environ["AUTOGEN_WORKSPACE"], exist_ok=True)

# === Model and Tokenizer ===
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
hf_token = os.environ.get("HF_TOKEN")
openai_api_key = os.environ.get("OPENAI_API_KEY")
model_id = "ChienChung/my-llama-1b"
config_path = hf_hub_download(repo_id=model_id, filename="config.json", use_auth_token=hf_token, cache_dir="/tmp/huggingface")
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
if "rope_scaling" in config_dict:
    config_dict["rope_scaling"] = {"type": "dynamic", "factor": config_dict["rope_scaling"].get("factor", 32.0)}
model_config = LlamaConfig.from_dict(config_dict)
model_config.trust_remote_code = True

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_auth_token=hf_token, cache_dir="/tmp/huggingface")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Initialize OpenAI client ===
openai = OpenAI(api_key=openai_api_key)

# === AutoGen setup ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Helper: file path ===
def get_file_path(upload):
    if isinstance(upload, str):
        return upload
    data = upload.get("data")
    name = upload.get("name", "file")
    td = tempfile.mkdtemp()
    path = os.path.join(td, name)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, (bytes, bytearray)) else data.encode())
    return path

# === Function definitions for OpenAI function calling ===

def fn_multi_doc_qa(arguments: dict) -> dict:
    query = arguments.get("query")
    files = arguments.get("files", [])
    docs = []
    file_names = []
    for upload in files:
        path = get_file_path(upload)
        file_names.append(os.path.basename(path))
        if path.lower().endswith('.pdf'):
            loader = PyPDFLoader(path)
        elif path.lower().endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            loader = TextLoader(path)
        pages = loader.load()
        docs.extend(pages)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    # AutoGen multi-document analysis
    temp_dir = tempfile.mkdtemp(dir=os.environ["AUTOGEN_WORKSPACE"])
    os.chdir(temp_dir)
    user_proxy = UserProxyAgent(
        name="User", system_message="User multi-doc QA", human_input_mode="NEVER",
        code_execution_config={"use_docker": False}, llm_config={"model":"gpt-4o-mini","api_key":openai_api_key}
    )
    analyzer = AssistantAgent(name="DocAnalyzer", system_message="Analyze documents.", llm_config={"model":"gpt-4o-mini","api_key":openai_api_key})
    group = GroupChat(agents=[user_proxy, analyzer], messages=[], max_round=3)
    manager = GroupChatManager(groupchat=group, llm_config={"model":"gpt-4o-mini","api_key":openai_api_key})
    context = "\n\n".join(f"Document {n}: {p.page_content[:1000]}" for n,p in enumerate(docs))
    prompt = f"Query: {query}\nContext:\n{context}"
    user_proxy.initiate_chat(manager, message=prompt)
    result = user_proxy.last_message()["content"]
    return {"result": result}


def fn_summarize(arguments: dict) -> dict:
    query = arguments.get("query")
    files = arguments.get("files", [])
    # Simplest summarization: call multi_doc_qa with summarize intent
    resp = fn_multi_doc_qa({"query": "Summarise: "+query, "files": files})
    return {"summary": resp.get("result")}


def fn_doc_qa(arguments: dict) -> dict:
    query = arguments.get("query")
    file = arguments.get("file")
    path = get_file_path(file)
    if path.lower().endswith('.pdf'):
        loader = PyPDFLoader(path)
    elif path.lower().endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(path)
    else:
        loader = TextLoader(path)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    # use simple LLM QA
    content = "\n".join(d.page_content for d in chunks)
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": f"Answer: {query}\nContext:\n{content[:2000]}"}]
    )
    return {"answer": completion.choices[0].message.content}


def fn_math(arguments: dict) -> dict:
    expr = arguments.get("expression")
    import numexpr as ne
    try:
        result = ne.evaluate(expr)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def fn_time(arguments: dict) -> dict:
    from datetime import datetime
    now = datetime.utcnow()
    return {"utc_time": now.isoformat()+"Z"}


def fn_weather(arguments: dict) -> dict:
    query = arguments.get("query")
    api_key = os.environ.get("WEATHER_API_KEY")
    # call external WeatherAPI
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={query}"
    data = requests.get(url).json()
    return {"weather": data}


def fn_search(arguments: dict) -> dict:
    query = arguments.get("query")
    # placeholder: return query
    return {"results": f"Searched for {query}"}

# Function specs for OpenAI
function_specs = [
    {
        "name": "multi_doc_qa",
        "description": "Answer queries across multiple documents",
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
            }
        },
        "required": ["query"]
    },
    {
        "name": "summarize",
        "description": "Summarise documents",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "files": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            }
        },
        "required": ["query"]
    },
    {
        "name": "doc_qa",
        "description": "Answer on single document",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "file": {"type": "object"}
            }
        },
        "required": ["query", "file"]
    },
    {
        "name": "math",
        "description": "Calculate expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            }
        },
        "required": ["expression"]
    },
    {
        "name": "time",
        "description": "Get current time",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "weather",
        "description": "Get weather for location",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        },
        "required": ["query"]
    },
    {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        },
        "required": ["query"]
    }
]

# Map function name to implementation
fn_map = {
    "multi_doc_qa": fn_multi_doc_qa,
    "summarize": fn_summarize,
    "doc_qa": fn_doc_qa,
    "math": fn_math,
    "time": fn_time,
    "weather": fn_weather,
    "search": fn_search
}

# === FastAPI endpoint ===
app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_msg = body.get("query")
    files = body.get("files", [])
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": user_msg}],
        functions=function_specs,
        function_call="auto"
    )
    msg = response.choices[0].message
    if msg.get("function_call"):
        name = msg["function_call"]["name"]
        args = json.loads(msg["function_call"]["arguments"])
        # attach uploads if needed
        if name in ["multi_doc_qa","summarize"]:
            args["files"] = files
        if name == "doc_qa" and files:
            args["file"] = files[0]
        result = fn_map[name](args)
        return JSONResponse(content={"result": result})
    else:
        return JSONResponse(content={"result": msg.get("content")})

# === Run Uvicorn ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
