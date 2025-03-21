# app.py
import os
import torch
import gradio as gr
import time

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import QAWithSourcesChain
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# 1) 選擇 LLaMA 模型
model_id = "meta-llama/Llama-3.2-1B-Instruct"
hf_token = os.environ.get("HF_TOKEN", None)

# 2) 判斷裝置
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device => {device}")

# 3) 載入模型 & tokenizer
model_config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=hf_token,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=model_config,
    trust_remote_code=True,
    token=hf_token,
)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=hf_token,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4) 建立 text-generation pipeline
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_length=1024,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

# 5) 包裝成 HuggingFacePipeline，供 LangChain 的 LLM 使用
llm = HuggingFacePipeline(pipeline=llama_pipeline)

# 6) 載入檔案 & 向量化
# 假設 data/ 下有 biden-sotu-2023-planned-official.txt
file_path = "data/biden-sotu-2023-planned-official.txt"
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", 
                                   model_kwargs={"token": hf_token})
vectordb = Chroma.from_documents(all_splits, embeddings, persist_directory="chroma_db")
retriever = vectordb.as_retriever()

# 7) 建立 RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

def rag_query(user_query):
    """對使用者輸入進行 RAG 查詢並回傳結果"""
    start_time = time.time()
    result = qa.run(user_query)
    end_time = time.time()
    return f"Answer (in {round(end_time - start_time, 2)}s):\n\n{result}"

# 8) Gradio 介面
def gradio_interface(query):
    return rag_query(query)

demo = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="RAG with LLaMA",
    description="Enter your question about Biden's SOTU 2023."
)

if __name__ == "__main__":
    demo.launch()