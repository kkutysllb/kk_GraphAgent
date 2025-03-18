"""
主应用入口
提供Flask Web服务，处理自然语言查询请求
"""
import os
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio

from src.agents.base import Message
from src.agents.orchestrator import OrchestratorAgent
from src.utils.config import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agno.log")
    ]
)

logger = logging.getLogger("agno.app")

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域请求支持

# 获取配置
config_path = os.environ.get("AGNO_CONFIG", "config.json")
config = load_config(config_path)

# 创建编排器
orchestrator = OrchestratorAgent(config)
logger.info("系统初始化完成，已加载编排器")

# 会话存储
sessions = {}

@app.route("/api/query", methods=["POST"])
async def process_query():
    """
    处理自然语言查询API
    """
    start_time = time.time()
    
    try:
        # 获取请求数据
        data = request.json
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400
        
        query = data.get("query")
        if not query:
            return jsonify({"error": "查询文本不能为空"}), 400
        
        # 获取或创建会话ID
        session_id = data.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"创建新会话: {session_id}")
        
        # 创建消息
        msg = Message(
            session_id=session_id,
            data={
                "query": query,
                "context": data.get("context", {}),
                "options": data.get("options", {})
            }
        )
        
        # 通过编排器处理消息
        result = await orchestrator.on_message(msg)
        
        # 检查结果
        if result.has_error():
            logger.warning(f"处理查询时出错: {result.error}")
            return jsonify({
                "status": "error",
                "error": result.error,
                "session_id": session_id,
                "processing_time": time.time() - start_time
            }), 500
        
        # 构建响应
        response = {
            "status": "success",
            "session_id": session_id,
            "processing_time": result.data.get("processing_time", time.time() - start_time),
            "result": _prepare_result(result.data)
        }
        
        # 更新会话
        sessions[session_id] = {
            "last_query": query,
            "last_result": result.data,
            "last_updated": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"处理查询时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": f"服务器错误: {str(e)}",
            "processing_time": time.time() - start_time
        }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    健康检查接口
    """
    return jsonify({
        "status": "ok",
        "version": config.get("version", "1.0.0"),
        "system": "Agno Graph Agent",
        "time": time.time()
    })

def _prepare_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备API响应结果，移除内部数据
    """
    # 复制数据以避免修改原始数据
    result = data.copy()
    
    # 移除可能的内部数据
    internal_keys = [
        "error", "schema_raw", "embedding", "vector", 
        "session_data", "traceback", "raw_response"
    ]
    
    for key in internal_keys:
        if key in result:
            del result[key]
    
    # 确保结果包含必要字段
    if "cypher_query" in result:
        result["cypher"] = result.pop("cypher_query")
    
    return result

async def main():
    """
    异步主函数
    """
    # 可以在这里进行额外的异步初始化
    pass

if __name__ == "__main__":
    # 获取端口
    port = int(os.environ.get("PORT", 5000))
    
    # 执行异步初始化
    asyncio.run(main())
    
    # 启动Flask应用
    logger.info(f"启动Web服务器，监听端口: {port}")
    app.run(host="0.0.0.0", port=port) 