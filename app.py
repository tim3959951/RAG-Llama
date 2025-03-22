import os
import time
import shutil
import json
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from huggingface_hub import hf_hub_download
import gradio as gr


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device => {device}")

hf_token = os.environ.get("HF_TOKEN")
model_id = "meta-llama/Llama-3.2-1B-Instruct"

config_path = hf_hub_download(repo_id=model_id, filename="config.json", use_auth_token=hf_token)
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

if "rope_scaling" in config_dict:
    config_dict["rope_scaling"] = {
        "type": "dynamic",
        "factor": config_dict["rope_scaling"].get("factor", 32.0)
    }

model_config = LlamaConfig.from_dict(config_dict)
model_config.trust_remote_code = True

print("Loading Llama model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=model_config,
    trust_remote_code=True,
    use_auth_token=hf_token,
)
model.to(device)
print("Model loaded!")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_auth_token=hf_token,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded!")

prompt = "Explain AI in one sentence:"
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generation result =>", generated_text)

device_map = None if device == "cpu" else "auto"
query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device_map,
    max_length=1024
)

print("Pipeline ready.")


try:
    if not os.path.exists("/tmp/chroma_db"):
        print("Copying prebuilt chroma_db to /tmp/chroma_db ...")
        shutil.copytree("./chroma_db", "/tmp/chroma_db")

    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(
        persist_directory="/tmp/chroma_db",
        embedding_function=embeddings
    )
    print("Chroma loaded successfully.")

    retriever = vectordb.as_retriever()

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful AI assistant. Use only the text from the context below to answer the user's question.
If the answer is not in the context, say "No relevant info found."
Return only the final answer in one to three sentences.
Do not restate the question or context.
Do not include these instructions in your final output.
Context:
{context}

Question: 
{question}

Answer:
"""
    )

    # HuggingFacePipeline from langchain
    from langchain.llms import HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=query_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    print("RetrievalQA chain created.")

    def rag_qa(user_query):
        raw_output = qa.run(user_query)
        lower_text = raw_output.lower()
        split_token = "answer:"
        idx = lower_text.find(split_token)
        if idx != -1:  
            final_answer = raw_output[idx + len(split_token) :].strip()
            return final_answer
        else:
            return raw_output
    demo_description = """
    **Context**:
    This demo is powered by a Retrieval-Augmented Generation (RAG) approach using 
    Biden’s 2023 State of the Union Address as the primary document. 
    All answers are derived from that transcript. 
    If the answer is not in the text, the system should respond with "No relevant info found."

    **Sample Questions**:
    1. What were the main topics regarding infrastructure in this speech?
    2. How does the speech address the competition with China?
    3. What does Biden say about job growth in the past two years?
    4. Does the speech mention anything about Social Security or Medicare?
    5. What does the speech propose regarding Big Tech or online privacy?

    Feel free to ask any question relevant to Biden’s 2023 State of the Union Address.
    """
    demo = gr.Interface(
        fn=rag_qa,
        inputs="text",
        outputs="text",
        title="Biden 2023 SOTU RAG QA Demo",
        description=demo_description,
        allow_flagging="never" 
    )

except Exception as e:
    print("[ERROR] Something went wrong in Step3:", e)

    def fallback_inference(user_query):
        return "We encountered an error loading RetrievalQA. Only direct Llama inference is available."

    demo = gr.Interface(
        fn=fallback_inference,
        inputs="text",
        outputs="text",
        title="Biden 2023 SOTU RAG QA Demo",
        description="No retrieval available due to error.",
        allow_flagging="never" 
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
