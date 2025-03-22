# RAG-LLaMA: Retrieval-Augmented Generation with LLaMA

This repository showcases a **Retrieval-Augmented Generation (RAG)** pipeline using a **quantised Large Language Model (LLaMA)** and **Chroma** as the vector database. The system answers questions about **President Biden’s 2023 State of the Union Address (SOTU)** by retrieving relevant text chunks (custom documents can be substituted), then generating grounded answers.

While LLMs are powerful at understanding context and generating responses, they may hallucinate when asked about unseen information. RAG mitigates this by combining an external retriever (based on vector search via text embeddings) with a generator (LLM). The interaction between both components is orchestrated using **LangChain**.

This project also demonstrates **deployment** [here](https://huggingface.co/spaces/ChienChung/RAG-Llama3) for real-time inference.


---

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Future Improvements](#future-improvements)
6. [License](#license)

---

## Overview

**Objective:**  
Develop a retrieval-augmented question answering system tailored to President Biden’s 2023 State of the Union Address by integrating a fine-tuned LLaMA model with an efficient vector database.

**Key Highlights:**
- **Llama-3.2-1B-Instruct Model:** A smaller, fine-tuned variant of the LLaMA family optimized for text generation.
- **Chroma Vector Database:** Stores text embeddings for rapid and relevant retrieval.
- **LangChain Integration:** Uses RetrievalQA (chain_type="stuff") to blend retrieved context with model inference.
- **Gradio Demo:** Offers a user-friendly web interface for interactive queries on Hugging Face Spaces.

---

## System Architecture

1. **Model Loading:**
   - The system downloads the `meta-llama/Llama-3.2-1B-Instruct` model from Hugging Face, applies necessary configuration patches, and initializes it on the device.

2. **Embedding & Retrieval:**
   - The complete text of the 2023 State of the Union Address is segmented into chunks, embedded, and stored in a local Chroma database. When a query is received, the system retrieves the most relevant segments via `Chroma.as_retriever()`.

3. **Retrieval-Augmented Generation (RAG):**
   - LangChain’s RetrievalQA leverages the retrieved text fragments as context for the LLaMA model, which then generates a context-aware final answer.

4. **Gradio Interface:**
   - A Gradio interface (default port 7860) is launched, allowing users to input queries and receive immediate answers.
     
---

## Project Structure

| File/Folder                                | Description                                      |
|--------------------------------------------|--------------------------------------------------|
| 📂 `chorma_db`                         | Chroma DB with SOTU text embeddings.             |
| 📄 `app.py`                               | Main script for model loading, retrieval, and UI.  |
| 📄 `requirements.txt`                     | Required Python dependencies.                   |
| 📄 `Dockerfile`                           | Docker setup and `app.py` execution.             |
| 📄 `biden-sotu-2023-planned-official.txt` | Full SOTU text as the knowledge base.            |
| 📄 `rag-llama3.2-langchain-chromadb.ipynb` | Training pipeline notebook.                      |
| 📄 `README.md`                   | This file.                           |
| 📄 `.gitignore`                  | List of files/folders to ignore.                 |


---

## How to Run
#### Online: [Hugging Face Spaces](https://huggingface.co/spaces/ChienChung/RAG-Llama3)
### Local Setup
1. **Clone** the repo and enter the directory:
   ```bash
   git clone https://github.com/tim3959951/RAG-Llama.git
   cd rag-llama
   ```
2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Export** your Hugging Face token (see **Note** below):
   ```bash
   export HF_TOKEN=your_token_here  # macOS/Linux
   ```
4. **Run** the app:
   ```bash
   python app.py
   ```
   - By default, it launches a Gradio interface at http://127.0.0.1:7860.
     
### Using Docker
1. **Clone** the repo and enter the directory:
   ```bash
   git clone https://github.com/tim3959951/RAG-Llama.git
   cd rag-llama
   ```
2. **Build** the image:
   ```bash
   docker build -t rag-llama .
   ```
3. **Run** the container:
   ```bash
   docker run -p 7860:7860 -e HF_TOKEN=your_token_here rag-llama
   ```
   - Visit http://localhost:7860 to interact with the demo.

> **Note**: This project used a gated model, you need appropriate Hugging Face credentials. Make sure to [request access](https://huggingface.co/meta-llama/) and include your token.
>
> <details>
> <summary>To set up the credentials and taken:</summary>
> 
> 1. **Request access to the model** here: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
> 2. **Log in and generate your access token**: https://huggingface.co/settings/tokens Create a token with `read` permissions.
> 3. **Export your token to the environment** before running the app:   
>
>   On Max/Linux:
>   ```bash
>   export HF_TOKEN=your_token_here
>   ```
>   On Windows:
>   ```bash
>   set HF_TOKEN=your_token_here
>   ```
>   </details>
  
### Example Questions
- “What were the main topics regarding infrastructure in this speech?”
- “How does the speech address the competition with China?”
- “What does Biden say about Social Security or Medicare?”
  
---

## Future Improvements

- **Scaling:** Experiment with larger LLaMA models or explore QLoRA fine-tuning to enhance performance.
- **Online RAG:** Integrate real-time web scraping or additional knowledge bases to dynamically update the context.
- **UI Enhancements:** Improve the Gradio interface with richer features such as usage metrics and advanced query handling.

---

## License

This project is distributed under the terms specified in the repository’s license file. For details on the LLaMA base model usage, please refer to [Meta’s model license](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and the associated terms on Hugging Face.

---

**Thank you for exploring RAG-LLaMA!**  
For any questions, issues, or collaboration opportunities, please open an issue or connect via [LinkedIn](https://www.linkedin.com/in/tim-cch).
Feel free to open issues or pull requests, or connect on [LinkedIn](https://www.linkedin.com/in/tim-cch).  
