#!/usr/bin/env python
import os
import shutil
import tempfile
import json
import torch
import re
import requests
import transformers
import chardet
import deepeval
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from huggingface_hub import hf_hub_download
from typing import List, Dict, Any
import gradio as gr
from pathlib import Path


# Solve permission issues
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/tmp/huggingface/metrics"
os.environ["GRADIO_FLAGGING_DIR"] = "/tmp/flagged"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/sentence_transformers"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hf_cache"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# 設置環境變數，確保 AutoGen 可以寫入臨時目錄
os.environ["AUTOGEN_WORKSPACE"] = "/tmp/autogen_workspace"
os.makedirs("/tmp/autogen_workspace", exist_ok=True)
os.chmod("/tmp/autogen_workspace", 0o777)  # 確保目錄可寫

# 設置 OpenAI API 相關環境變數
os.environ["OPENAI_API_TYPE"] = "open_ai"  # 如果您使用的是 OpenAI API



#  建立 temp 安全區
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["DEEPEVAL_RESULTS_FOLDER"] = "/tmp/deepeval_results"
os.makedirs("/tmp/deepeval_results", exist_ok=True)

#  修正 Python tempdir 基底（避免它寫 home）
import tempfile
tempfile.tempdir = "/tmp"
# 在此處加入 DeepEval 的 monkey-patch，避免全域更改工作目錄
original_evaluate = deepeval.evaluate

def patched_evaluate(*args, **kwargs):
    current_dir = os.getcwd()
    try:
        os.chdir("/tmp")
        return original_evaluate(*args, **kwargs)
    finally:
        os.chdir(current_dir)

deepeval.evaluate = patched_evaluate


SHOW_EVAL = os.getenv("SHOW_EVAL", "false").lower() == "true"


# Load Required Modules 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from tempfile import mkdtemp
from langchain.schema import AIMessage
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dateutil import parser as date_parser
import numexpr as ne
import pandas as pd

# Multi-Agent Imports 
from serpapi import GoogleSearch
# CrewAI Section: completely use CrewAI's Agent, Task, Crew and @tool decorator
from crewai import Crew, Agent, Task, Process
from crewai.tools import tool
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from langchain_experimental.agents import create_pandas_dataframe_agent
from langsmith import traceable
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
# from langgraph.graph import Graph
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from sentence_transformers import SentenceTransformer
# === AutoGen for multi-intent collaboration ===
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager



try:
    from phoenix.trace.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument()
except Exception as e:
    print(f"[WARNING] Failed to load Phoenix LangChain trace: {e}")

session_retriever = None
session_qa_chain = None
csv_dataframe = None  # CSV tool will use this

# Safe Result Formatter 
def safe_format_result(result) -> str:
    try:
        if hasattr(result, "agent_name") and hasattr(result, "output"):
            return f"[Agent: {result.agent_name}]\n{result.output}"
        elif isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return json.dumps(result, indent=2)
        elif isinstance(result, list):
            return "\n".join(str(r) for r in result)
        else:
            return str(result)
    except Exception as e:
        return f"Error formatting result: {e}"


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

api_app = FastAPI()

@api_app.post("/multi_agent_chat")
async def multi_agent_chat(request: Request):
    data = await request.json()
    query = data.get("query")
    result = multi_agent_chat_advanced(query)
    return JSONResponse(content={"result": result})

@api_app.post("/multi_doc_qa")
async def multi_doc_qa(request: Request):
    data = await request.json()
    query = data.get("query")
    result = langgraph_tab6_main(query)
    return JSONResponse(content={"result": result})
    
# Model and Device Setup 
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device => {device}")

hf_token = os.environ.get("HF_TOKEN")
openai_api_key = os.environ.get("OPENAI_API_KEY")
model_id = "ChienChung/my-llama-1b"

config_path = hf_hub_download(
    repo_id=model_id,
    filename="config.json",
    use_auth_token=hf_token,
    cache_dir="/tmp/huggingface"
)
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
if "rope_scaling" in config_dict:
    config_dict["rope_scaling"] = {"type": "dynamic", "factor": config_dict["rope_scaling"].get("factor", 32.0)}
model_config = LlamaConfig.from_dict(config_dict)
model_config.trust_remote_code = True



print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_auth_token=hf_token,
    cache_dir="/tmp/huggingface"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded!")


# Chroma DB and Document Retrieval Setup 
print("Loading Chroma DB for Biden Speech...")
if not os.path.exists("/tmp/chroma_db"):
    shutil.copytree("./chroma_db", "/tmp/chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory="/tmp/chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever()

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant. Use only the text from the context below to answer the user's question.
If the answer is not in the context, say "No relevant info found."
If the question is not in the context, say "No relevant info found."

Return only the final answer in one to three sentences.
Do not restate the question or context.
Do not include these instructions in your final output.

Context:
{context}

Question: {question}

Answer:
"""
)


llm_gpt4 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)
crew_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=openai_api_key
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_gpt = ConversationalRetrievalChain.from_llm(
    llm=llm_gpt4,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# Helper Function: Extract file path from uploaded file
def get_file_path(file):
    if isinstance(file, str):
        return file
    elif isinstance(file, dict):
        # Prefer using the "data" key, then "name"
        return file.get("data", file.get("name", None))
    elif hasattr(file, "save"):
        temp_dir = mkdtemp()
        file_path = os.path.join(temp_dir, file.name)
        file.save(file_path)
        return file_path
    else:
        return None

# Original functionalities (Tabs 1-4) functions

@traceable(name="GPT-4 Document QA")
def rag_gpt4_qa(query):
    raw_answer = qa_gpt.run(query)

    if SHOW_EVAL:
        try:
            top_docs = retriever.get_relevant_documents(query)
            test_case = LLMTestCase(
                input=query,
                actual_output=raw_answer,
                expected_output=raw_answer,
                context=[doc.page_content for doc in top_docs[:3]]
            )
            metric = AnswerRelevancyMetric(model="gpt-4o-mini")
            results = evaluate([test_case], [metric])
            result = results[0]
            print(f"[DeepEval Tab4] Input: {query}")
            print(f"[DeepEval Tab4] Passed: {result.passed}, Score: {result.score:.2f}, Reason: {result.reason}")
        except Exception as e:
            print(f"[DeepEval Tab4] Evaluation failed: {e}")

    return raw_answer

@traceable(name="Upload Document QA")
def upload_and_chat(file, query):
    file_path = get_file_path(file)
    if file_path is None:
        return "Unable to obtain the uploaded file path."
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    temp_retriever = db.as_retriever()
    qa_temp = RetrievalQA.from_chain_type(
        llm=llm_gpt4,
        chain_type="stuff",
        retriever=temp_retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    raw_answer = qa_temp.run(query)
    if SHOW_EVAL:
        try:
            test_case = LLMTestCase(
                input=query,
                actual_output=raw_answer,
                expected_output=raw_answer,
                context=[d.page_content for d in chunks[:3]]
            )
            metric = AnswerRelevancyMetric(model="gpt-4o-mini")  # default is GPT-4o
            results = evaluate([test_case], [metric])
            result = results[0]
            print(f"[DeepEval QA] Input: {query}")
            print(f"[DeepEval QA] Passed: {result.passed}, Score: {result.score:.2f}, Reason: {result.reason}")
        except Exception as e:
            print(f"[DeepEval QA] Evaluation failed: {e}")

    return raw_answer

   
initial_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Write a concise and structured summary of the following content. Focus on capturing the main ideas and key details:

{text}

--- Summary ---
"""
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template="""You already have an existing summary:
{existing_answer}

Refine the summary based on the new content below. Add or update information only if it's relevant. Keep it concise:

{text}

--- Refined Summary ---
"""
)

@traceable(name="Document Summarise")
def document_summarize(file):
    file_path = get_file_path(file)
    if file_path is None:
        return "Unable to obtain the uploaded file."
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()
    summarize_chain = load_summarize_chain(llm_gpt4, chain_type="refine", question_prompt=initial_prompt, refine_prompt=refine_prompt)
    summary = summarize_chain.invoke(docs)
    return summary['output_text']

def csv_agent(file, query):
    file_path = get_file_path(file)
    if file_path is None:
        return "Unable to obtain the uploaded CSV file."
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        df = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        return f"Error reading CSV: {e}"
    safe_dict = {"df": df}
    try:
        result = ne.evaluate(query, local_dict=safe_dict)
        return str(result)
    except Exception as e:
        return f"Query error: {e}"

def search_web(query):
    if isinstance(query, dict):
        query = query.get("query", "")
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "SERPAPI_API_KEY not set. Please set the environment variable."
    params = {"engine": "google", "q": query, "api_key": api_key, "num": 10}
    search = GoogleSearch(params)
    results = search.get_dict()
    if "organic_results" in results:
        raw_output = ""
        for result in results["organic_results"]:
            title = result.get("title", "No Title")
            link = result.get("link", "No Link")
            snippet = result.get("snippet", "No Snippet")
            raw_output += f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n\n"
        prompt = f"""
You are a helpful assistant. Given the following web search results and the user's question:

1. First, identify **only** the most relevant entries.
2. Then, summarise the key insights using fluent, **British English**.
3. If the user's question asks for trends, categories, or multiple items, you may present key points in a non-markdown bullet-point style (e.g. use "•" instead of "-", avoid using "**" for bold).
4. Otherwise, reply in a short, natural paragraph.
5. Do **not** use any markdown formatting such as `**`, `##`, or list syntax.
6. Keep your answer concise and professional, as if explaining to a colleague.
7. If no relevant info is found, say: "Sorry, I couldn't find a reliable answer from the current results."

--- Web Search Results ---
{raw_output}

--- User's Question ---
"{query}"

Answer:
"""
        summarized = _general_chat(prompt)
        return summarized if summarized else raw_output.strip()
    else:
        return "No results found."

def uploaded_qa(file, query):
    file_path = get_file_path(file)
    if file_path is None:
        return "Unable to obtain the uploaded file path."
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    temp_retriever = db.as_retriever()
    qa_temp = RetrievalQA.from_chain_type(
        llm=llm_gpt4,
        chain_type="stuff",
        retriever=temp_retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa_temp.run(query)

# CrewAI Multi-Agent System (Tab 5) 
# Completely abandon langchain.agents.Tool and use CrewAI's @tool decorator to define tools
from pydantic import BaseModel
class SimpleQuery(BaseModel):
    query: str

def _general_chat(query: str) -> str:
    try:
        response = llm_gpt4.invoke(query)
        if isinstance(response, AIMessage):
            response = response.content  # Extract the actual string
        if any(kw in response.lower() for kw in ["i'm not sure", "i don't know", "no information", "can't find"]):
            return _search_web_tool(query)
        return response
    except Exception as e:
        return f"General chat error: {e}"
@tool("general_chat")
def general_chat_tool(query: str) -> str:
    """General assistant: Answer general questions without relying on documents."""
    try:
        response = llm_gpt4.invoke(query)
        if isinstance(response, AIMessage):
            response = response.content  # Extract the actual string
        if any(kw in response.lower() for kw in ["i'm not sure", "i don't know", "no information", "can't find"]):
            return search_web(query)
        return response
    except Exception as e:
        return f"General chat error: {e}"

def location_to_timezone(location: str) -> str:
    try:
        geo = Nominatim(user_agent="time_agent_demo")
        loc = geo.geocode(location)
        if not loc:
            return "Europe/London"
        tf = TimezoneFinder()
        return tf.timezone_at(lng=loc.longitude, lat=loc.latitude) or "Europe/London"
    except Exception:
        return "Europe/London"
        
def get_time_tool(query: str) -> str:

    # use GPT to find location keyword
    try:
        location_prompt = f"""
        You are a location extractor. Given a user's query about time or date, return the location mentioned in it. If not found, return "London".

        Examples:
        - "What's the time in Tokyo now?" → Tokyo
        - "今天台北幾點？" → Taipei
        - "現在在紐約幾點？" → New York
        - "今天幾號？" → London
        - "What date is today？" → London

        Now process this query: "{query}"
        """
        location_response = llm_gpt4.invoke(location_prompt)
        if isinstance(location_response, AIMessage):
            location = location_response.content.strip()
        else:
            location = str(location_response).strip()
    except Exception as e:
        location = "London"

    location_key = location.lower()
    tz_str = location_to_timezone(location)
    now = datetime.now(ZoneInfo(tz_str))

    # return time or date
    q_lower = query.lower()
    if any(k in q_lower for k in ["date", "幾號", "today", "day"]):
        return now.strftime(f"The date in {location.title()} is %B %d, %Y (%A).")
    elif any(k in q_lower for k in ["time", "幾點", "現在"]):
        return now.strftime(f"The time in {location.title()} is %I:%M %p.")
    else:
        return now.strftime(f"The local time in {location.title()} is %I:%M %p on %B %d, %Y.")

@tool("time_tl")
def time_tool(query: str) -> str:
    """Time Agent: Answer time or date queries worldwide using LLM + GeoLocator + TimezoneFinder."""
    # use GPT to find location keyword
    try:
        location_prompt = f"""
        You are a location extractor. Given a user's query about time or date, return the location mentioned in it. If not found, return "London".

        Examples:
        - "What's the time in Tokyo now?" → Tokyo
        - "今天台北幾點？" → Taipei
        - "現在在紐約幾點？" → New York
        - "今天幾號？" → London
        - "What date is today？" → London

        Now process this query: "{query}"
        """
        location_response = llm_gpt4.invoke(location_prompt)
        if isinstance(location_response, AIMessage):
            location = location_response.content.strip()
        else:
            location = str(location_response).strip()
    except Exception as e:
        location = "London"

    location_key = location.lower()
    tz_str = location_to_timezone(location)
    now = datetime.now(ZoneInfo(tz_str))

    # return time or date
    q_lower = query.lower()
    if any(k in q_lower for k in ["date", "幾號", "today", "day"]):
        return now.strftime(f"The date in {location.title()} is %B %d, %Y (%A).")
    elif any(k in q_lower for k in ["time", "幾點", "現在"]):
        return now.strftime(f"The time in {location.title()} is %I:%M %p.")
    else:
        return now.strftime(f"The local time in {location.title()} is %I:%M %p on %B %d, %Y.")

weather_api_key = os.environ.get("WEATHER_API_KEY")


def get_time_tool2(query: str) -> datetime:
    try:
        # Step 1: 抽出地點
        location_prompt = f"""
        You are a location extractor. Given a user's query about time or date, return the location mentioned in it.
        If not found, return "London".

        Query: "{query}"
        """
        location_response = llm_gpt4.invoke(location_prompt)
        location = location_response.content.strip() if isinstance(location_response, AIMessage) else str(location_response).strip()

        # Step 2: 當地目前時間（加入 DEBUG）
        print(f"[DEBUG] Extracted Location: {location}")
        tz_str = location_to_timezone(location)
        print(f"[DEBUG] Timezone: {tz_str}")
        now = datetime.now(ZoneInfo(tz_str))
        print(f"[DEBUG] Local Time at {location}: {now}")

        # Step 3: 動態 few-shot prompt（每次更新 based on now）
        examples = [
            ("five hours later", now + timedelta(hours=5)),
            ("later", now + timedelta(hours=2)),
            ("soon", now + timedelta(minutes=30)),
            ("shortly", now + timedelta(minutes=15)),
            ("after a while", now + timedelta(hours=1)),
            ("tomorrow at 3pm", now.replace(hour=15, minute=0, second=0) + timedelta(days=1)),
            ("the day after tomorrow at 10am", now.replace(hour=10, minute=0, second=0) + timedelta(days=2)),
            ("last Monday 9am", (now - timedelta(days=(now.weekday() + 7))).replace(hour=9, minute=0, second=0)),
            ("next Monday", (now + timedelta(days=(7 - now.weekday()))).replace(hour=12, minute=0, second=0)),
            ("last Friday", (now - timedelta(days=(now.weekday() - 4 + 7) % 7)).replace(hour=12, minute=0, second=0)),
            ("next Friday", (now + timedelta(days=(4 - now.weekday() + 7) % 7)).replace(hour=12, minute=0, second=0)),
            ("in 10 hours", now + timedelta(hours=10)),
            ("this weekend", (now + timedelta(days=(5 - now.weekday()) % 7)).replace(hour=10, minute=0, second=0)),
            ("next weekend", (now + timedelta(days=((5 - now.weekday()) % 7) + 7)).replace(hour=10, minute=0, second=0)),
            ("下週一下午三點", (now + timedelta(days=(7 - now.weekday() + 0) % 7)).replace(hour=15, minute=0, second=0)),
            ("昨天下午五點", (now - timedelta(days=1)).replace(hour=17, minute=0, second=0)),
            ("昨天早上八點", (now - timedelta(days=1)).replace(hour=8, minute=0, second=0)),
            ("later this evening", now.replace(hour=20, minute=0, second=0)),
            ("現在", now),
            ("last month", (now - timedelta(days=30)).replace(hour=12, minute=0, second=0)),
            ("early tomorrow morning", now.replace(hour=6, minute=0, second=0) + timedelta(days=1)),
            ("in 2 hours", now + timedelta(hours=2)),
            ("in one hour", now + timedelta(hours=1)),
            ("in 30 minutes", now + timedelta(minutes=30)),
            ("in a few minutes", now + timedelta(minutes=10)),
        ]

        # 加入 local time 說明在 Examples 區段
        examples_header = f"""Assume the current local time in {location} is exactly:
**{now.strftime('%Y-%m-%d %H:%M:%S')}** (timezone: {tz_str})

Use this exact time to reason all examples below.
"""
        examples_str = "\n".join([f'User Query: "{q}" → {dt.strftime("%Y-%m-%d %H:%M:%S")}' for q, dt in examples])

        # Step 4: 构建完整 prompt
        # Step 4: 构建完整 prompt
        time_query_prompt = f"""
You are a timezone-aware time reasoner. Based on the user's query, calculate the **exact target time** they are referring to.
Remember: all relative expressions like "later", "in 2 hours", "tomorrow" must be strictly calculated based on the current local time above.
{examples_header}

Please return the result in this **exact format**: `YYYY-MM-DD HH:MM:SS` (24-hour clock, no timezone info).
Only return the time string — no explanation, no extra words.

### Examples:
{examples_str}

### Now process:
User Query: "{query}"
→
"""

        time_response = llm_gpt4.invoke(time_query_prompt)
        time_str = time_response.content.strip() if isinstance(time_response, AIMessage) else str(time_response).strip()

        # Step 5: 嘗試解析時間
        try:
            target_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            target_time = target_time.replace(tzinfo=ZoneInfo(tz_str))
            return target_time
        except Exception:
            return f"Failed to parse time string from LLM: '{time_str}'"

    except Exception as e:
        return f"Error in retrieving location or time information: {e}"

        
def weather_agent_tool(query: str) -> str:
    """Weather Agent: Return current, hourly, or historical weather info using WeatherAPI."""
    try:
        weather_api_key = os.environ.get("WEATHER_API_KEY")
        if not weather_api_key:
            return "Weather API key not found. Please set WEATHER_API_KEY env variable."

        # Step 1: Extract location
        location_prompt = f"""
        You are a location extractor. Given a user's query about weather, extract the location mentioned in it.
        If not found, return "London".

        Examples:
        - "Is it gonna rain in Tokyo?" → Tokyo
        - "Will it be hot in New York later?" → New York
        - "明天下午高雄會不會下雨？" → Kaohsiung
        - "How’s the weather?" → London

        Query: "{query}"
        """
        location_resp = llm_gpt4.invoke(location_prompt)
        location = location_resp.content.strip() if isinstance(location_resp, AIMessage) else str(location_resp).strip()

        # Step 2: Get timezone and time
        target_dt = get_time_tool2(query)

       # if isinstance(target_dt, str):
       #     target_dt = datetime.strptime(target_dt, "%Y-%m-%d %H:%M:%S")
        if not isinstance(target_dt, datetime):
            return f"Failed to parse the target time from your query. Got: {target_dt}"
        
        tz_str = location_to_timezone(location)
        target_dt = target_dt.replace(tzinfo=ZoneInfo(tz_str))
        now = datetime.now(ZoneInfo(tz_str))  # 用同一時區的 now 去比較！

        # Step 3: Check limits and decide API
        if target_dt < now - timedelta(days=7):
            return "Only supports up to 7 days of historical data."
        elif target_dt > now + timedelta(days=2):
            return "Only supports up to 3 days of forecast."

        if target_dt < now:
            url = f"http://api.weatherapi.com/v1/history.json?key={weather_api_key}&q={location}&dt={target_dt.strftime('%Y-%m-%d')}"
        else:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={weather_api_key}&q={location}&days=3&aqi=no&alerts=no"

        data = requests.get(url).json()
        forecast_hours = []
        if "forecast" in data:
            for day in data["forecast"]["forecastday"]:
                for hour in day["hour"]:
                    forecast_hours.append(hour)
        elif "forecastday" in data:
            forecast_hours = data["forecastday"][0]["hour"]
        else:
            return "No forecast data available."

        # Step 4: Find closest hour
        min_diff = float("inf")
        closest_hour = None
        for hour_data in forecast_hours:
            hour_dt = date_parser.parse(hour_data["time"]).replace(tzinfo=ZoneInfo(tz_str))
            diff = abs((hour_dt - target_dt).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_hour = hour_data

        if not closest_hour:
            return f"No hourly data found for {target_dt.strftime('%Y-%m-%d %H:%M')}."

        # Step 5: Generate summary
        condition = closest_hour["condition"]["text"]
        temp = closest_hour["temp_c"]
        feels = closest_hour["feelslike_c"]
        humidity = closest_hour["humidity"]
        chance_rain = closest_hour.get("chance_of_rain", 0)
        hour_str = closest_hour["time"].split(" ")[1]

        summary_prompt = f"""
You are a helpful weather reasoning assistant.

The user wants to know about the weather conditions at a specific time: {target_dt.strftime('%Y-%m-%d %H:%M')} in {location}.  
Use the data below to answer their question. This may refer to the past, present, or future — do not assume it is the current weather.

Based on the following weather data and the user's question, think step-by-step to extract the most relevant information, and give a natural, friendly, and cautious answer in British English.

Avoid being overly confident — never say "Yes, it will..." or "Definitely." Instead, use expressions like:
- "It is very likely that..."
- "There is a high chance of..."
- "Based on the available data, it seems that..."
- "There may be..."

Also, after answering the question, include a short weather summary and a useful suggestion (e.g., bring an umbrella, wear sunscreen, avoid outdoor activities).

**Do not use markdown formatting such as `*`, `**`, or list symbols.**

--- Weather Data ---
Location: {location}
Time: {target_dt.strftime('%Y-%m-%d')} at {hour_str}
Condition: {condition}
Temperature: {temp}°C (Feels like {feels}°C)
Humidity: {humidity}%
Chance of rain: {chance_rain}%
Chance of snow: {closest_hour.get("chance_of_snow", "N/A")}%
Wind speed: {closest_hour.get("wind_kph", "N/A")} kph
UV index: {closest_hour.get("uv", "N/A")}
Cloud cover: {closest_hour.get("cloud", "N/A")}%
Visibility: {closest_hour.get("vis_km", "N/A")} km

--- User Question ---
{query}

--- Final Answer ---
"""
        response = llm_gpt4.invoke(summary_prompt)
        return response.content.strip() if isinstance(response, AIMessage) else str(response)

    except Exception as e:
        return f"Weather Agent Error: {e}"
        

@tool("weather")
def weather_tool(query: str) -> str:
    """Weather Agent: Return current, hourly, or historical weather info using WeatherAPI."""
    try:
        weather_api_key = os.environ.get("WEATHER_API_KEY")
        if not weather_api_key:
            return "Weather API key not found. Please set WEATHER_API_KEY env variable."

        # Step 1: Extract location
        location_prompt = f"""
        You are a location extractor. Given a user's query about weather, extract the location mentioned in it.
        If not found, return "London".

        Examples:
        - "Is it gonna rain in Tokyo?" → Tokyo
        - "Will it be hot in New York later?" → New York
        - "明天下午高雄會不會下雨？" → Kaohsiung
        - "How’s the weather?" → London

        Query: "{query}"
        """
        location_resp = llm_gpt4.invoke(location_prompt)
        location = location_resp.content.strip() if isinstance(location_resp, AIMessage) else str(location_resp).strip()

        # Step 2: Get timezone and time
        target_dt = get_time_tool2(query)

       # if isinstance(target_dt, str):
       #     target_dt = datetime.strptime(target_dt, "%Y-%m-%d %H:%M:%S")
        if not isinstance(target_dt, datetime):
            return f"Failed to parse the target time from your query. Got: {target_dt}"

        tz_str = location_to_timezone(location)
        target_dt = target_dt.replace(tzinfo=ZoneInfo(tz_str))
        now = datetime.now(ZoneInfo(tz_str))  # 用同一時區的 now 去比較！

        # Step 3: Check limits and decide API
        if target_dt < now - timedelta(days=7):
            return "Only supports up to 7 days of historical data."
        elif target_dt > now + timedelta(days=2):
            return "Only supports up to 3 days of forecast."

        if target_dt < now:
            url = f"http://api.weatherapi.com/v1/history.json?key={weather_api_key}&q={location}&dt={target_dt.strftime('%Y-%m-%d')}"
        else:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={weather_api_key}&q={location}&days=3&aqi=no&alerts=no"

        data = requests.get(url).json()
        forecast_hours = []
        if "forecast" in data:
            for day in data["forecast"]["forecastday"]:
                for hour in day["hour"]:
                    forecast_hours.append(hour)
        elif "forecastday" in data:
            forecast_hours = data["forecastday"][0]["hour"]
        else:
            return "No forecast data available."

        # Step 4: Find closest hour
        min_diff = float("inf")
        closest_hour = None
        for hour_data in forecast_hours:
            hour_dt = date_parser.parse(hour_data["time"]).replace(tzinfo=ZoneInfo(tz_str))
            diff = abs((hour_dt - target_dt).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_hour = hour_data

        if not closest_hour:
            return f"No hourly data found for {target_dt.strftime('%Y-%m-%d %H:%M')}."

        # Step 5: Generate summary
        condition = closest_hour["condition"]["text"]
        temp = closest_hour["temp_c"]
        feels = closest_hour["feelslike_c"]
        humidity = closest_hour["humidity"]
        chance_rain = closest_hour.get("chance_of_rain", 0)
        hour_str = closest_hour["time"].split(" ")[1]

        summary_prompt = f"""
You are a helpful weather reasoning assistant.

The user wants to know about the weather conditions at a specific time: {target_dt.strftime('%Y-%m-%d %H:%M')} in {location}.  
Use the data below to answer their question. This may refer to the past, present, or future — do not assume it is the current weather.

Based on the following weather data and the user's question, think step-by-step to extract the most relevant information, and give a natural, friendly, and cautious answer in British English.

Avoid being overly confident — never say "Yes, it will..." or "Definitely." Instead, use expressions like:
- "It is very likely that..."
- "There is a high chance of..."
- "Based on the available data, it seems that..."
- "There may be..."

Also, after answering the question, include a short weather summary and a useful suggestion (e.g., bring an umbrella, wear sunscreen, avoid outdoor activities).

**Do not use markdown formatting such as `*`, `**`, or list symbols.**

--- Weather Data ---
Location: {location}
Time: {target_dt.strftime('%Y-%m-%d')} at {hour_str}
Condition: {condition}
Temperature: {temp}°C (Feels like {feels}°C)
Humidity: {humidity}%
Chance of rain: {chance_rain}%
Chance of snow: {closest_hour.get("chance_of_snow", "N/A")}%
Wind speed: {closest_hour.get("wind_kph", "N/A")} kph
UV index: {closest_hour.get("uv", "N/A")}
Cloud cover: {closest_hour.get("cloud", "N/A")}%
Visibility: {closest_hour.get("vis_km", "N/A")} km

--- User Question ---
{query}

--- Final Answer ---
"""
        response = llm_gpt4.invoke(summary_prompt)
        return response.content.strip() if isinstance(response, AIMessage) else str(response)

    except Exception as e:
        return f"Weather Agent Error: {e}"
        
@tool("summarise")
def summarise_tool(query: str) -> str:
    """Summarise: Use document summarisation functionality."""
    global session_retriever, session_qa_chain
    if session_retriever is None:
        return "No document uploaded."
    try:
        docs = session_retriever.get_relevant_documents(query if query.strip() else "summary")
        if not docs:
            return "No relevant content found in the document."
        summarize_chain = load_summarize_chain(llm_gpt4, chain_type="refine", question_prompt=initial_prompt, refine_prompt=refine_prompt)
        summary = summarize_chain.invoke(docs)
        return summary['output_text']
    except Exception as e:
        return f"Summarisation error: {e}"
        
def _calc_tool(query: str) -> str:
    import math
    import re
    try:
        # Handle pure arithmetic expressions (only numbers and symbols)
        if re.fullmatch(r"[0-9\.\+\-\*/%\^\(\)\s]+", query.strip()):
            cleaned = query.strip().replace("^", "**")
            result = ne.evaluate(cleaned)
            return f"The result is: {result}"

        # For expressions containing sin/cos/log etc., automatically apply math + radians
        expr = query.lower()
        expr = re.sub(r'sin\(([^)]+)\)', r'sin(math.radians(\1))', expr)
        expr = re.sub(r'cos\(([^)]+)\)', r'cos(math.radians(\1))', expr)
        expr = re.sub(r'tan\(([^)]+)\)', r'tan(math.radians(\1))', expr)
        expr = expr.replace("^", "**")

        result = eval(expr, {"__builtins__": None}, {
            "math": math, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log10, "sqrt": math.sqrt, "exp": math.exp,
            "pi": math.pi, "e": math.e
        })
        return f"The result is: {result}"
    
    except Exception:
        try:
            # Fallback: ask GPT to calculate and explain briefly in plain English (avoid messy symbols)
            response = llm_gpt4.invoke(f"Please calculate this and explain briefly in plain English: {query}. Avoid math symbols like $ or \\n or \\(.")
            result = response.content if isinstance(response, AIMessage) else response
            result = re.sub(r"\\\[.*?\\\]", "", result)  # Remove \[...\]
            result = re.sub(r"\\\(.*?\\\)", "", result)  # Remove \(...\)
            return result.strip()
        except Exception as e:
            return f"Natural language fallback error: {e}"
        
@tool("python_calc")
def python_calc_tool(query: str) -> str:
    """Python Calculation: Perform basic arithmetic or logical operations."""
    try:
        result = ne.evaluate(query)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"
def _search_web_tool(query: str) -> str:
    return search_web(query)
@tool("search_tool")
def search_tool_func(query: str) -> str:
    """Search: Perform web searches using external search engines."""
    return search_web(query)

@tool("uploaded_qa")
def uploaded_qa_tool_func(query: str) -> str:
    """Document QA: Answer questions based on the uploaded document content."""
    global session_qa_chain
    if session_qa_chain is not None:
        try:
            return session_qa_chain.run(query)
        except Exception as e:
            return f"Document QA error: {e}"
    else:
        return "No document uploaded."
        
@tool("csv_agent")
def csv_tool_func(query: str) -> str:
    """CSV Agent: Use natural language to analyse uploaded CSV files."""
    global csv_dataframe
    if csv_dataframe is None:
        return "No CSV file uploaded."
    try:
        agent = create_pandas_dataframe_agent(llm=llm_gpt4, df=csv_dataframe, verbose=True)
        return agent.run(f"Here is the table:\n{csv_dataframe.head().to_string(index=False)}\n\n{query}")
    except Exception as e:
        return f"CSV Agent error: {e}"

# Establish CrewAI agents (for Tab 5 only)
general_agent = Agent(
    role="General Assistant",
    goal="Respond to any general query that is not related to documents or CSV files.",
    backstory="You're an intelligent assistant who answers questions about anything general, such as math, dates, or general knowledge.",
    tools=[general_chat_tool],
    verbose=True
)
summarizer_agent = Agent(
    role="Document Summarizer",
    goal="Summarise the content of the uploaded document.",
    backstory="You are a professional summarisation expert who can identify key points in long documents.",
    tools=[summarise_tool],
    verbose=True
)
document_qa_agent = Agent(
    role="Document QA Specialist",
    goal="Answer questions based on the uploaded document.",
    backstory="You are an expert in document understanding and can accurately extract answers.",
    tools=[uploaded_qa_tool_func],
    verbose=True
)

search_agent = Agent(
    role="Search Expert",
    goal="Search the web and provide relevant information.",
    backstory="You are an expert at finding relevant information from the internet.",
    tools=[search_tool_func],
    verbose=True
)
time_agent = Agent(
    role="Time Assistant",
    goal="Answer current time or date related questions across different time zones.",
    backstory="You're a time-aware agent who can tell time or date in any major city.",
    tools=[time_tool],
    verbose=True
)

weather_agent = Agent(
    role="Weather Expert",
    goal="Answer global weather queries.",
    backstory="You are a weather analyst who provides accurate and real-time weather information for any location.",
    tools=[weather_tool],
    verbose=True
)

math_agent = Agent(
    role="Math Assistant",
    goal="Perform accurate arithmetic or logical calculations.",
    backstory="You are a calculator expert skilled at quick computations.",
    tools=[python_calc_tool],
    verbose=True
)
csv_agent = Agent(
    role="CSV Analyst",
    goal="Analyse tabular data and answer questions about the uploaded CSV file.",
    backstory="You are skilled in interpreting tabular datasets and can extract numerical or logical insights.",
    tools=[csv_tool_func],
    verbose=True
)
router_agent = Agent(
    role="Query Router",
    goal="Determine the most suitable agent or tool to handle the user query.",
    backstory="You are an intelligent query dispatcher that analyses the user's intent and chooses the best AI agent to answer.",
    tools=[python_calc_tool, search_tool_func, csv_tool_func, uploaded_qa_tool_func, summarise_tool, general_chat_tool, time_tool, weather_tool],
    verbose=True
)
router_task = Task(
    description="""
Based on the user's query, decide which agent or tool is best suited to handle it:
- If the query is related to the content of an uploaded file (e.g., 'what is this document about?'), send it to the **Document QA Agent**.
- If the query contains words like 'summarize', 'summary', or 'main points', use the **Summarizer Agent**.
- If the query **includes any numbers or symbols** (like +, -, *, /, %, ^), or **mentions math terms** (like 'calculate', 'how much', 'percent', 'square root', 'log', 'cos', 'sin', etc.), or starts with 'what is', 'what’s', 'how much is', assume it is a **math question** and send it to the **Math Agent**.
- If the user uploaded a CSV file and asks about table content, data trends, or uses words like 'data', 'table', 'csv', 'column', or 'row', send it to the **CSV Agent**.
- If the user asks about current events, trending topics, or online information (e.g., 'What is LangChain?', 'latest news'), send it to the **Search Agent**.
- If the query is about current date, time, or day of week (e.g., 'what is today's date?', 'what time is it?', 'what day is it?', '現在幾點', '今天幾號', '禮拜幾'), send it to the **Time Agent**.
- If the query is about weather, rain, temperature, or forecasts (e.g., "What's the weather in Paris?", "Will it rain tomorrow in London?"), send it to the **Weather Agent**.
- If the question is general and not related to documents, calculations, CSVs, or the internet (e.g., 'Who are you?', 'Tell me a fun fact'), send it to the **General Agent**.
- If none of these apply, use your best judgment to choose the most relevant agent.
""",
    expected_output="The final answer from the selected agent or tool.",
    agent=router_agent,
    input_variables=["query"]
)

crew = Crew(
    agents=[general_agent, summarizer_agent, document_qa_agent, search_agent, math_agent, time_agent, csv_agent, weather_agent],
    tasks=[router_task],
    process=Process.sequential,
    verbose=True,
    llm=crew_llm
)

# test qa
def build_langgraph_doc_qa_chain(llm, retriever, memory, prompt):
    def retrieve_step(state):
        docs = state['retriever'].get_relevant_documents(state['query'])
        return {"docs": docs, **state}

    def answer_step(state):
        prompt = state["prompt"]
        llm = state["llm"]
        docs = state["docs"]

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        doc_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )
    # 只執行一次，並傳入所有需要的參數
        answer = doc_chain.run({
            "input_documents": docs,
            "question": state["query"]
        })
        return {"answer": answer, **state}

    builder = StateGraph(dict)
    builder.add_node("Retrieve", retrieve_step)
    builder.add_node("Answer", answer_step)
    builder.set_entry_point("Retrieve")
    builder.add_edge("Retrieve", "Answer")
    builder.set_finish_point("Answer")

    compiled = builder.compile()

    def run(query):
        return compiled.invoke({
            "query": query,
            "retriever": retriever,
            "llm": llm,
            "prompt": prompt
        })

    return run

@traceable(name="Multi-Agent Chat")
def multi_agent_chat_advanced(query: str, file=None) -> str:
    global session_retriever, session_qa_chain, csv_dataframe

    # Smart routing without needing uploaded files 
    lower_query = query.lower()

    math_keywords = ["how much", "calculate", "what is", "what’s", "%", "sin", "cos", "log", "sqrt", "^", "*", "/", "+", "-", "="]
    if any(k in lower_query for k in math_keywords):
        return _calc_tool(query)

    date_keywords = ["what date", "today", "what time", "what day", "current time", "date", "現在幾點", "今天幾號", "禮拜幾"]
    if any(k in lower_query for k in date_keywords):
        return get_time_tool(query)
    weather_keywords = ["weather", "rain", "snow", "cold", "hot", "sunscreen", "sunglasses", "umbrella", "windy", "cloudy", "sunny", "temperature", "forecast", "天氣", "會不會下雨", "冷嗎", "熱嗎", "氣溫"]
    if any(k in lower_query for k in weather_keywords):
        return weather_agent_tool(query)
    search_keywords = ["latest", "news", "startup", "startups", "company", "companies", "top", "trending", "in 2025", "in 2024", "tell me"]
    if any(k in lower_query for k in search_keywords):
        return search_web(query)

    general_keywords = ["who are you", "what is your name", "what can you do", "fun fact"]
    if any(k in lower_query for k in general_keywords):
        return _general_chat(query)

    # Check if file exists and determine its format 
    file_path = get_file_path(file) if file is not None else None

    # Determine if the query should be processed as document-related
    non_doc_keywords = ["calculate", "sum", "date", "time", "how many", "how much", "weather", "temperature"]
    use_file_chain = not any(kw in query.lower() for kw in non_doc_keywords)

    # Step 3: If a file is uploaded 
    if file_path:
        file_lower = file_path.lower()

        # Process CSV 
        if file_lower.endswith(".csv"):
            try:
                with open(file_path, 'rb') as f:
                    result = chardet.detect(f.read())
                    encoding = result['encoding']
                df = pd.read_csv(file_path, encoding=encoding)
                csv_dataframe = df  # Ensure global assignment

                # If query mentions file, add context
                if "file" in query.lower() or "upload" in query.lower():
                    query = f"The user uploaded the following CSV file:\n\n{query}"

                result = crew.kickoff(inputs={"query": query})
                return safe_format_result(result)
            except Exception as e:
                return f"CSV Parsing Error: {e}"

        # 3-2: Process PDF / DOCX / TXT
        elif file_lower.endswith((".pdf", ".txt", ".docx")):
            try:
                loader = (
                    PyPDFLoader(file_path) if file_lower.endswith(".pdf")
                    else UnstructuredWordDocumentLoader(file_path) if file_lower.endswith(".docx")
                    else TextLoader(file_path)
                )
                docs = loader.load()
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
                db = FAISS.from_documents(chunks, embeddings)
                session_retriever = db.as_retriever()
                session_qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm_gpt4,
                    retriever=session_retriever,
                    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
                )

                # If the query is summary-related, use Summarize Chain
                if any(kw in query.lower() for kw in ["summarize", "summary", "summarise", "summarisation", "summarization", "摘要", "總結"]):
                    return document_summarize(file_path)

                # If using QA Chain is appropriate
                if use_file_chain:
                    try:
                        answer = session_qa_chain.run(query)
                        #session_graph_chain = build_langgraph_doc_qa_chain(llm_gpt4, session_retriever, memory, custom_prompt)
                        #answer = session_graph_chain(query)["answer"]

                        # ✅ DeepEval 評估僅在 Tab1 文件 QA 的情況下觸發
                        if SHOW_EVAL:
                            try:
                                test_case = LLMTestCase(
                                    input=query,
                                    actual_output=answer,
                                    expected_output=answer,
                                    context=[d.page_content for d in session_retriever.get_relevant_documents(query)[:3]]
                                )
                                metric = AnswerRelevancyMetric(model="gpt-4o-mini")
                                results = evaluate([test_case], [metric])
                                result = results[0]
                                print(f"[DeepEval Tab1] Input: {query}")
                                print(f"[DeepEval Tab1] Passed: {result.passed}, Score: {result.score:.2f}, Reason: {result.reason}")
                            except Exception as e:
                                print(f"[DeepEval Tab1] Evaluation failed: {e}")

                        return answer
                    except Exception as e:
                        return f"Document QA Error: {e}"

                # Otherwise, proceed with Multi-Agent reasoning
                if "file" in query.lower() or "upload" in query.lower():
                    query = f"The user uploaded the following document:\n\n{query}"

                result = crew.kickoff(inputs={"query": query})
                return safe_format_result(result)

            except Exception as e:
                return f"Document Processing Error: {e}"

        else:
            return "Unsupported file format."

    # Step 4: If no file is uploaded, directly use CrewAI reasoning 
    try:
        result = crew.kickoff(inputs={"query": query})
        return safe_format_result(result)
    except Exception as e:
        return f"Multi-Agent Error: {e}"



# LangGraph 使用的節點函數（會接續你的 Crew Agent）
# 初始化 embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Intent Embedding 分類（支援檔名）
INTENT_LABELS = {
    "DocQA": ["document", "file", "paper", "cb", "proposal", "project"],
    "Summarise": ["summarise", "summary", "abstract", "key points", "overview", "main points"],
    "General": ["who are you", "tell me something", "what can you do", "fun fact"],
}

def parse_query(query: str) -> dict:
    prompt = """Analyze the following query and determine required subtasks. Return a JSON object containing:
    - summarize_files: list of document indices to summarize
    - qa_pairs: list of QA objects [{"question": "question", "doc_indices": [relevant doc indices]}]
    - compare_files: list of document index pairs to compare [[doc1_idx, doc2_idx]]
    - find_relations: boolean, whether to analyze document relationships
    
    For example, query "What are the differences between document A and B, and summarize A" should return:
    {
        "summarize_files": [0],
        "qa_pairs": [],
        "compare_files": [[0, 1]],
        "find_relations": false
    }
    
    Query: """ + query
    
    response = llm_gpt4.invoke(prompt)
    try:
        return json.loads(response.content)
    except:
        return {
            "summarize_files": [],
            "qa_pairs": [{"question": query, "doc_indices": [0]}],
            "compare_files": [],
            "find_relations": False
        }


def autogen_multi_document_analysis(query: str, docs: list, file_names: list) -> str:
    try:
        # 建立絕對路徑的暫存目錄，並確保它存在
        import tempfile
        import os
        
        # 建立一個臨時工作目錄
        temp_dir = tempfile.mkdtemp(dir="/tmp")
        os.environ["OPENAI_CACHE_DIR"] = temp_dir
        
        # 設置 AutoGen 的工作目錄
        os.environ["AUTOGEN_CACHE_PATH"] = temp_dir
        os.environ["AUTOGEN_CACHEDIR"] = temp_dir
        os.environ["OPENAI_CACHE_PATH"] = temp_dir
        
        # 強制 AutoGen 使用我們的臨時目錄而不是 ./.cache
        import autogen
        if hasattr(autogen, "set_cache_dir"):
            autogen.set_cache_dir(temp_dir)
        
        # 準備文件上下文
        context = "\n\n".join(
            f"Document {name}:\n{doc[:2000]}..." 
            for name, doc in zip(file_names, docs)
        )

        # 配置 LLM
        config_list = [{
            "model": "gpt-4o-mini",
            "api_key": openai_api_key
        }]

        # 基礎配置 - 不包含任何緩存相關參數
        llm_config = {
            "config_list": config_list,
            "temperature": 0
        }
        
        # 在進行 AutoGen 處理前，切換到臨時目錄
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # 以下是您的 AutoGen 處理代碼
            user_proxy = UserProxyAgent(
                name="User",
                system_message="A user seeking information from multiple documents.",
                human_input_mode="NEVER",
                code_execution_config={"use_docker": False},
                llm_config=llm_config
            )



            # 定義文檔分析專家
            doc_analyzer = AssistantAgent(
                name="DocumentAnalyzer",
                system_message="""You are an expert at analyzing and comparing documents. Focus on:
                1. Key similarities and differences
                2. Main themes and topics
                3. Relationships between documents
                4. Evidence-based analysis""",
                llm_config=llm_config
            )

            # 定義問答專家
            qa_expert = AssistantAgent(
                name="QAExpert",
                system_message="""You are an expert at extracting specific information. Focus on:
                1. Finding relevant details
                2. Answering specific questions
                3. Cross-referencing information
                4. Providing evidence""",
                llm_config=llm_config
            )

            # 定義總結專家
            summarizer = AssistantAgent(
                name="Summarizer",
                system_message="""You are an expert at summarizing content. Focus on:
                1. Key points and findings
                2. Important relationships
                3. Critical conclusions
                4. Comprehensive overview""",
                llm_config=llm_config
            )

            # 創建群組聊天
            groupchat = GroupChat(
                agents=[user_proxy, doc_analyzer, qa_expert, summarizer],
                messages=[],
                max_round=5
            )

            # 創建管理器
            manager = GroupChatManager(
                groupchat=groupchat,
                llm_config=llm_config
            )

            # 準備任務提示
            task_prompt = f"""Analyze these documents and answer the query:
        
            Query: {query}

            Documents Context:
            {context}

            Requirements:
            1. Provide a direct and clear answer
            2. Support all claims with evidence from the documents
            3. Consider relationships between all documents
            4. If comparing, analyze all relevant aspects
            5. If summarizing, cover all important points
            6. If looking for specific content, search thoroughly
            7. If analyzing relationships, consider all connections

            Please provide a comprehensive and well-structured answer."""

            # 執行群組討論
            user_proxy.initiate_chat(manager, message=task_prompt)
            return user_proxy.last_message()["content"]
        finally:
            # 完成後，切回原始目錄
            os.chdir(original_dir)
            
        return result

    except Exception as e:
        print(f"ERROR in AutoGen processing: {str(e)}")
        return f"Error analyzing documents: {str(e)}"





    
# === AutoGen 多代理人協作邏輯 ===

        
def detect_intent_embedding(query, file_names=[]):
    query_emb = embedding_model.encode(query, normalize_embeddings=True)
    best_label = None
    best_score = -1
    all_phrases = INTENT_LABELS.copy()
    if file_names:
        all_phrases["DocQA"] += [name.lower() for name in file_names]
    for label, examples in all_phrases.items():
        for example in examples:
            example_emb = embedding_model.encode(example, normalize_embeddings=True)
            score = float(query_emb @ example_emb.T)
            if score > best_score:
                best_score = score
                best_label = label
    return best_label if best_label else "General"

def decide_next(state):
    query = state.get("query", "")
    file_names = state.get("file_names", [])
    label = detect_intent_embedding(query, file_names)
    return label

# === 定義 Task 物件 ===
docqa_task = Task(
    description="Document QA Task: Answer questions based on the uploaded document.",
    expected_output="Answer from Document QA Agent.",
    agent=document_qa_agent,
    input_variables=["query"]
)

general_task = Task(
    description="General Chat Task: Answer general queries.",
    expected_output="Answer from General Agent.",
    agent=general_agent,
    input_variables=["query"]
)

summariser_task = Task(
    description="Summarisation Task: Summarise document content.",
    expected_output="Summary output.",
    agent=summarizer_agent,  # 注意此處名稱須與定義一致（使用字母 z）
    input_variables=["query"]
)

search_task = Task(
    description="Search Task: Retrieve information from the web.",
    expected_output="Answer from Search Agent.",
    agent=search_agent,
    input_variables=["query"]
)

# === LangGraph 節點函數 ===

def general_run(state):
    """改用直接 LLM 回答取代 General Agent"""
    try:
        prompt = f"""You are a helpful AI assistant. Please answer the following question:
        {state["query"]}
        
        Provide a clear and informative answer."""
        
        response = llm_gpt4.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        return {"answer": answer}
    except Exception as e:
        print(f"ERROR in general_run: {str(e)}")
        return {"answer": "I apologize, but I'm having trouble processing your request."}


def docqa_run(state):
    """文件問答處理"""
    try:
        # 如果有檢索器，使用檢索器
        if "retriever" in state:
            relevant_docs = state["retriever"].get_relevant_documents(state["query"])
            context = "\n".join(d.page_content for d in relevant_docs)
        else:
            context = "\n".join(state["docs"])
            
        prompt = f"""Based on the following context, please answer the question:
        Question: {state["query"]}
        
        Context:
        {context[:3000]}
        
        Provide a detailed and accurate answer based on the context."""
        
        response = llm_gpt4.invoke(prompt)
        return {"answer": response.content if hasattr(response, 'content') else str(response)}
    except Exception as e:
        print(f"ERROR in docqa_run: {str(e)}")
        return general_run(state)


def summariser_run(state):
    """文件摘要處理"""
    try:
        context = "\n".join(state["docs"])
        prompt = f"""Please provide a comprehensive summary of the following document:
        {context[:3000]}
        
        Focus on:
        1. Main topics and key points
        2. Important findings or conclusions
        3. Significant details"""
        
        response = llm_gpt4.invoke(prompt)
        return {"summary": response.content if hasattr(response, 'content') else str(response)}
    except Exception as e:
        print(f"ERROR in summariser_run: {str(e)}")
        return {"summary": "Error generating summary."}

# === LangGraph 定義 ===
def build_langgraph_pipeline():
    graph = StateGraph(dict)
    graph.add_node("Router", lambda state: state)  # Router 僅傳遞狀態
    graph.add_node("DocQA", docqa_run)
    graph.add_node("General", general_run)
    graph.add_node("Summarise", summariser_run)
    graph.set_entry_point("Router")
    graph.add_conditional_edges("Router", decide_next, {
        "DocQA": "DocQA",
        "General": "General",
        "Summarise": "Summarise",
    })
    graph.set_finish_point("DocQA")
    graph.set_finish_point("General")
    graph.set_finish_point("Summarise")
    return graph.compile()

from tempfile import mkdtemp

def get_file_path_tab6(file):
    if isinstance(file, str):
        print("DEBUG: File is a string:", file)
        if os.path.exists(file):
            print("DEBUG: File exists:", file)
            return file
        else:
            print("DEBUG: File does not exist:", file)
            return None
    elif isinstance(file, dict):
        print("DEBUG: File is a dict:", file)
        data = file.get("data")
        name = file.get("name")
        print("DEBUG: Data:", data, "Name:", name)
        if data:
            if isinstance(data, str) and os.path.exists(data):
                print("DEBUG: Data is a valid file path:", data)
                return data
            else:
                temp_dir = mkdtemp()
                file_path = os.path.join(temp_dir, name if name else "uploaded_file")
                print("DEBUG: Writing data to temporary file:", file_path)
                with open(file_path, "wb") as f:
                    if isinstance(data, str):
                        f.write(data.encode("utf-8"))
                    else:
                        f.write(data)
                if os.path.exists(file_path):
                    print("DEBUG: Temporary file created:", file_path)
                    return file_path
                else:
                    print("ERROR: Temporary file not created:", file_path)
                    return None
        else:
            print("DEBUG: No data in dict, returning None")
            return None
    elif hasattr(file, "save"):
        print("DEBUG: File has save attribute")
        temp_dir = mkdtemp()
        file_path = os.path.join(temp_dir, file.name)
        file.save(file_path)
        if os.path.exists(file_path):
            print("DEBUG: File saved to:", file_path)
            return file_path
        else:
            print("ERROR: File not saved properly:", file_path)
            return None
    else:
        print("DEBUG: File type unrecognized")
        if hasattr(file, "name"):
            if os.path.exists(file.name):
                return file.name
        return None
        
def langgraph_tab6_main(query: str, file=None):
    try:
        print(f"DEBUG: Starting processing with query: {query}")
        
        # 如果沒有文件，直接使用 general_run
        if not file:
            return general_run({"query": query})["answer"]
        
        # 處理文件列表
        files = file if isinstance(file, list) else [file]
        all_docs = []
        file_names = []
        docs_by_file = []
        
        # 處理上傳的文件
        for f in files:
            try:
                path = get_file_path_tab6(f)
                if not path:
                    continue
                
                file_names.append(os.path.basename(path))
                
                # 根據文件類型選擇加載器
                if path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(path)
                elif path.lower().endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(path)
                else:
                    loader = TextLoader(path)
                
                docs = loader.load()
                if docs:
                    text = "\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))
                    docs_by_file.append(text)
                    all_docs.extend(docs)
            except Exception as e:
                print(f"ERROR processing file: {str(e)}")
                continue

        if not docs_by_file:
            return general_run({"query": query})["answer"]

        # 建立檢索器
        try:
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            ).split_documents(all_docs)
            
            db = FAISS.from_documents(chunks, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 5})
            
            global session_retriever, session_qa_chain
            session_retriever = retriever
            session_qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm_gpt4,
                retriever=retriever,
                memory=ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                ),
            )
        except Exception as e:
            print(f"ERROR setting up retriever: {str(e)}")
            retriever = None

        # 檢測是否為多文件查詢
        # 檢測是否為多文件查詢或複雜查詢
        if len(docs_by_file) > 1 or "compare" in query.lower() or "relation" in query.lower():
            return autogen_multi_document_analysis(query, docs_by_file, file_names)
            
        # 使用 LangGraph 處理單文件查詢
        state = {
            "query": query,
            "file_names": file_names,
            "docs": docs_by_file,
            "retriever": retriever
        }
        
        # 根據查詢意圖選擇處理方式
        intent = detect_intent_embedding(query, file_names)
        if intent == "Summarise":
            return summariser_run(state)["summary"]
        elif intent == "DocQA":
            return docqa_run(state)["answer"]
        else:
            return general_run(state)["answer"]
        
    except Exception as e:
        print(f"ERROR in main function: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"





# Gradio Interface Settings 
demo_description = """
**Context**:
This demo uses a **Retrieval-Augmented Generation (RAG)** system based on 
Biden’s 2023 State of the Union Address. 
All responses are grounded in this document. 
If no relevant information is found in the document, the system will say "No relevant info found."

**Sample Questions**:
1. What were the main topics regarding infrastructure in this speech?
2. How does the speech address the competition with China?
3. What does Biden say about job growth in the past two years?
4. Does the speech mention anything about Social Security or Medicare?
5. What does the speech propose regarding Big Tech or online privacy?

*Note: The LLaMA module generates responses based solely on the current query without follow-up memory or chat history management.*

> This is a CPU-only demo running a **quantised 1B LLaMA model**, built to show that full RAG + multi-agent systems can run even without a GPU. In production, the model can be replaced with larger ones (3B, 7B, etc.) and served using vLLM, 4-bit quantisation, or TensorRT for better speed. The design focuses on portability, deployment, and modularity.

Feel free to ask any question related to Biden’s 2023 State of the Union Address.
"""
demo_description2 = """
**Context**:
This demo uses a **Retrieval-Augmented Generation (RAG)** system based on 
Biden’s 2023 State of the Union Address. 
All responses are grounded in this document. 
If no relevant information is found in the document, the system will say "No relevant info found."

**Sample Questions**:
1. What were the main topics regarding infrastructure in this speech?
2. How does the speech address the competition with China?
3. What does Biden say about job growth in the past two years?
4. Does the speech mention anything about Social Security or Medicare?
5. What does the speech propose regarding Big Tech or online privacy?

*Note: The GPT module supports follow-up questions with conversation history management, enabling more interactive and context-aware discussions.*

Feel free to ask any question related to Biden’s 2023 State of the Union Address.
"""
demo_description3 = """
**Context**:
Upload a PDF, TXT, or DOCX file and ask a question about its content.
This demo uses **GPT-4o-Mini** to answer questions based on the content of your uploaded document.

Note: This is a **strict RAG-based QA** system. It will only answer questions if the answer is explicitly found in the uploaded document.
For more flexible or general-purpose responses, please try Tab 1 (Multi-Agent Assistant).

Typical Use Cases:
- Legal, technical, or academic documents where factual precision is critical
- Internal company reports where hallucination must be avoided
- Medical papers where only referenced content should be discussed

Feel free to ask any question directly related to your document.
"""
demo_description4 = """
**Context**:
This demo uses a **refinement-based document summarisation chain**.
Upload a PDF, TXT, or DOCX file to get a concise, structured summary of its contents.
"""
demo_description5 = """
**Context**:
This demo presents a GPT-style Multi-Agent AI Assistant, built with **LangChain, CrewAI**, and **RAG (Retrieval-Augmented Generation)**. The system automatically understands your intent and routes the query to the best expert agent, enabling dynamic **multi-agent orchestration**.

**Supported features**:
- 📄 **Document Summarisation** (PDF, DOCX, TXT)
- ❓ **FAQ-style Q&A based on uploaded documents** (RAG-based)
- 🌐 **Live Web Search** (Online RAG + GPT post-processing summary)
- 📅 **Real-time Worldwide Date & Time** (LLM + GeoLocator + TimezoneFinder, supports any city globally)
- 🌦️ **Global Weather** (LLM Time Reasoning + Timezone + Few-Shot, supports fuzzy queries, 3-day forecast, 7-day history, hourly precision)
- ➗ **Math and Logic Calculations** (from scientific equations to financial or tax-related use cases)
- 💬 **General Chatting / Reasoning**

**Sample Questions**:
1. Summarise the document. *(→ Summarisation Agent)*
2. What are the key ideas mentioned in this file? *(→ RAG QA Agent)*
3. What is LangChain used for? | What are the latest trends in AI startups in 2025? | Tell me the most recent breakthrough in quantum computing. *(→ Online Rag Agent)*
4. What's the current time in London? | What’s today’s date in New York? | What time is it in Taipei right now? *(→ Time Agent)*
5. Will it rain or snow in Sapporo tomorrow night? | Is it too windy for cycling in Amsterdam at 6am? | Do I need to bring an umbrella later this evening in Edinburgh? | Should I wear sunscreen in Bangkok around noon tomorrow? | Is it gonna rain later? | What was the weather like in Paris on last Sunday? | Will the weather be suitable for hiking at 3pm in Lake District? *(→ Weather Agent)*
6. If I earn $15 per hour and work 8 hours a day for 5 days, how much will I earn? | What is 5 * 22.5 / sin(45) | 3^3 + 4^2 | Calculate 25 * log(1000) | What is the square root of 144 *(→ Math Agent)*
7. Who are you? | What can you do? | What is the meaning of life? *(→ General Chat Agent)*

Feel free to upload a document and ask related questions, or just type a question directly—no file upload required. *Note: CSV file analysis and auto visualisation is coming soon.*
"""
demo_description6 = """
**Context**:
This is a **smart multi-document reasoning assistant**, powered by **LangGraph**, **CrewAI**, and **AutoGen**.
Upload zero to multiple files and ask anything — the system will uses **embedding-based intent detection** to decide whether to summarise, extract, compare, or analyse relationships.

For complex multi-file tasks, it triggers a **collaborative AutoGen team** to deeply reason across documents and generate contextual, evidence-based answers.

**Supported Features**:
- 📄 Multi-document support (PDF, DOCX, TXT)
- 🔍 Embedding-based intent detection and semantic routing
- 🤖 Agents: Summariser, QA Agent, General Agent, Search Agent
- 🔀 Orchestrated by LangGraph + AutoGen (fallbacks + task handoff)
- 🧠 AutoGen multi-agent collaboration for cross-file reasoning
- 🌐 Online search fallback if all the other agent can't handle tasks

**Sample Questions**:
1. Who are you? | What is GPT4? *(→ General Chat Agent)*
2. Summarise the document/file/your_doc_name. *(→ Summarisation Agent)*
3. What is LangChain used for? | What are the latest trends in AI startups in 2025? | Tell me the most recent breakthrough in quantum computing. *(→ Online Rag Agent)*
4. What's the title in the document? | What are the key ideas mentioned in this file? *(→ RAG QA Agent)*
5. Compare the proposals in DocA and DocB. | Summarise all files. | Is DocA one of the project in the DocB or DocC. | Which argument is stronger across these files? | Do these documents mention similar policies? | What's the difference between the files? *(→ AutoGen)*
6. What is LangChain used for? | What are the latest trends in AI startups in 2025? | Tell me the most recent breakthrough in quantum computing. *(→ Online Rag Agent)*

> Built for users who need clear, explainable, and context-aware answers — whether you’re working on documents in law, finance, research, or tech.
"""



demo = gr.TabbedInterface(
    interface_list=[
        gr.Interface(
            fn=langgraph_tab6_main,
            inputs=[
                gr.Textbox(label="Ask anything"),
                gr.File(label="Upload one or more files", file_types=[".pdf", ".txt", ".docx"], file_count="multiple")
            ],
            outputs="text",
            title="Smart Multi-Doc QA (LangGraph + AutoGen)",
            allow_flagging="never",
            description=demo_description6
        ),
        gr.Interface(
            fn=multi_agent_chat_advanced,
            inputs=[
                gr.Textbox(label="Enter your query"),
                gr.File(label="Upload file (CSV, PDF, TXT, DOCX)", file_types=[".pdf", ".txt", ".docx"], file_count="single")
            ],
            outputs="text",
            title="Multi-Agent AI Assistant",
            allow_flagging="never",
            description=demo_description5
        ),
        gr.Interface(
            fn=document_summarize,
            inputs=[gr.File(label="Upload PDF, TXT, or DOCX", file_types=[".pdf", ".txt", ".docx"])],
            outputs="text",
            title="Document Summarisation",
            allow_flagging="never",
            description=demo_description4
        ),
        gr.Interface(
            fn=upload_and_chat,
            inputs=[gr.File(label="Upload PDF, TXT, or DOCX", file_types=[".pdf", ".txt", ".docx"]), gr.Textbox(label="Ask a question")],
            outputs="text",
            title="Your Docs Q&A (Upload + GPT-4 RAG)",
            allow_flagging="never",
            description=demo_description3
        ),
        gr.Interface(
            fn=rag_gpt4_qa,
            inputs="text",
            outputs="text",
            title="Biden Q&A (GPT-4 RAG)",
            allow_flagging="never",
            description=demo_description2
        ),
        
    ],
    tab_names=[
        "Multi-Doc QA",
        "Multi-Agent AI Assistant",
        "Document Summarisation",
        "Your Docs Q&A (Upload + GPT-4 RAG)",
        "Biden Q&A (GPT-4 RAG)",
        "Biden Q&A (LLaMA RAG)",
        
    ],
    title="Smart RAG + Multi-Agent Assistant (with Web + Document AI)"
)

if __name__ == "__main__":
    import threading
    import uvicorn

    # 開 FastAPI 在另一個 thread
    def run_fastapi():
        uvicorn.run("app:api_app", host="0.0.0.0", port=8000)

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    # 同時跑 Gradio
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
