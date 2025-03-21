# 使用穩定的 Python 3.10 slim 作為基底映像檔
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 設定環境變數，將所有快取與配置路徑指向 /app 下可寫的目錄，
# 並延長 Hugging Face Hub 下載超時時間
ENV HOME=/app
ENV XDG_CACHE_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV MPLCONFIGDIR=/app/.config/matplotlib
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub
ENV HF_HUB_DOWNLOAD_TIMEOUT=60

# 建立快取、配置及 Hugging Face 快取目錄，並設定適當權限
RUN mkdir -p /app/cache /app/.config/matplotlib /app/.cache/huggingface/hub && \
    chmod -R 777 /app/cache /app/.config /app/.cache

# 安裝系統依賴（例如 git）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 到容器中
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案所有檔案到容器中
COPY . .

# 開放 Gradio 預設埠號 7860
EXPOSE 7860

# 啟動應用（會執行 app.py）
CMD ["python", "app.py"]
