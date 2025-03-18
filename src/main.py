"""
多智能体Cypher查询系统主程序
用于将自然语言查询转换为Neo4j Cypher查询
"""
import asyncio
import time
import uuid
import logging
import json
import os
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.base import Message, configure_message_bus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("agno.main")

# 创建FastAPI应用
app = FastAPI(
    title="Agno Cypher 转换系统",
    description="基于LLM的自然语言转Cypher查询多智能体协同解决方案",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class QueryRequest(BaseModel):
    query: str = Field(..., description="自然语言查询")
    session_id: Optional[str] = Field(None, description="会话ID")
    debug_mode: bool = Field(False, description="调试模式")

# 加载配置
def load_config() -> Dict[str, Any]:
    """加载系统配置"""
    config_path = os.environ.get('AGNO_CONFIG', 'config.json')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        # 返回默认配置
        return {
            "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
            "neo4j_password": os.environ.get("NEO4J_PASSWORD", "agno123"),
            "llm_endpoint": os.environ.get("LLM_ENDPOINT", "http://localhost:8080/v1"),
            "llm_model": os.environ.get("LLM_MODEL", "qwen2.5-7b"),
            "max_results": int(os.environ.get("MAX_RESULTS", "20")),
            "debug_mode": os.environ.get("DEBUG_MODE", "false").lower() == "true",
            "log_path": os.environ.get("LOG_PATH", "logs/agno.log"),
            "embedding_config": {
                "model_type": os.environ.get("EMBEDDING_MODEL_TYPE", "dual_stream"),
                "model_path": os.environ.get("EMBEDDING_MODEL_PATH", "models/dual_stream_encoder.pt")
            }
        }

# 加载配置
config = load_config()

# 配置消息总线和智能体
bus = configure_message_bus(config)

# 设置总管智能体的消息总线引用
from agents.orchestrator import OrchestratorAgent
orchestrator = bus.agents.get("orchestrator")
if isinstance(orchestrator, OrchestratorAgent):
    orchestrator.set_bus(bus)

@app.get("/")
async def root():
    """根路由，返回系统信息"""
    return {
        "name": "Agno Cypher 转换系统",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/v1/query")
async def process_query(request: QueryRequest):
    """处理自然语言查询，转换为Cypher并执行"""
    start_time = time.time()
    
    try:
        # 创建会话ID（如果未提供）
        session_id = request.session_id or f"session-{uuid.uuid4()}"
        
        # 创建消息
        msg = Message(
            data={"query": request.query},
            meta={"session_id": session_id}
        )
        
        # 通过总管智能体处理请求
        logger.info(f"处理查询: {request.query}")
        result = await bus.process("orchestrator", msg)
        
        # 检查是否有错误
        if "error" in result.data:
            logger.error(f"处理查询时出错: {result.data['error']}")
            raise HTTPException(status_code=400, detail=result.data["error"])
        
        # 构建响应
        response = {
            "query": request.query,
            "intent": result.data.get("intent", ""),
            "cypher": result.data.get("cypher", ""),
            "results": result.data.get("results", []),
            "execution_time": time.time() - start_time
        }
        
        # 添加分析结果（如果有）
        if "topology_analysis" in result.data:
            response["analysis"] = result.data["topology_analysis"]
        elif "performance_analysis" in result.data:
            response["analysis"] = result.data["performance_analysis"]
        elif "fault_diagnosis" in result.data:
            response["analysis"] = result.data["fault_diagnosis"]
        
        # 添加工作流跟踪（调试模式）
        if request.debug_mode or config.get("debug_mode", False):
            response["workflow_trace"] = result.workflow_results
        
        return response
        
    except Exception as e:
        logger.error(f"处理查询失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理查询失败: {str(e)}")

@app.get("/api/v1/schema")
async def get_schema():
    """获取数据库模式"""
    try:
        schema_msg = Message()
        result = await bus.process("schema_fetcher", schema_msg)
        
        if "error" in result.data:
            raise HTTPException(status_code=400, detail=result.data["error"])
            
        return result.data.get("schema", {})
        
    except Exception as e:
        logger.error(f"获取数据库模式失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取数据库模式失败: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    
    # 启动 Uvicorn 服务器
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true"
    ) 