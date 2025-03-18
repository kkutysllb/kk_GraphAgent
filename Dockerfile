FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt ./
COPY src/ ./src/
COPY config.json ./

# 创建必要的目录
RUN mkdir -p /app/cache /app/logs

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGNO_CONFIG=/app/config.json

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "src/app.py"] 