### 基于Agno框架的LLM自然语言转Cypher查询多智能体协同解决方案

---

#### **阶段1：基础设施搭建（1.5小时，已完成）**

```bash
# 1.1 使用Docker部署Neo4j 5.15
docker run -d --name neo4j_agno \
    -p 7474:7474 -p 7687:7687 \
    -v /data/neo4j:/data \
    -v /data/import:/var/lib/neo4j/import \
    -e NEO4J_AUTH=neo4j/agno123 \
    -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
    neo4j:5.12

# 1.2 Python环境配置（Python 3.10+）
conda create -n agno_cypher python=3.10
conda activate agno_cypher

# 1.3 安装核心依赖
pip install agno==0.9.3 
pip install "openai>=1.3" "py2neo>=2021.2" 
pip install "fastapi>=0.104" "uvicorn[standard]>=0.23"
pip install "sentence-transformers>=2.2.2" "pymilvus>=2.3.0"
pip install "torch>=2.0.0" "dgl>=1.1.0" "torch-geometric>=2.3.0"
```

---

#### **阶段2：图数据建模与初始化（数据已经完成导入neo4j，以下仅是示例代码）**

**2.1 领域模型设计（5G核心网资源图谱示例）**

```cypher
// 节点类型 - 5G核心网资源类型示例
CREATE (:VM {vmId: $id, name: $name, status: $status, cpu: $cpu, memory: $memory})
CREATE (:NE {neId: $id, name: $name, type: $type, status: $status})
CREATE (:HOST {hostId: $id, name: $name, status: $status, cpu: $cpu, memory: $memory})
CREATE (:DC {dcId: $id, name: $name, location: $location})

// 关系类型
CREATE (:DC)-[:CONTAINS]->(:HOST)
CREATE (:HOST)-[:DEPLOYS]->(:VM)
CREATE (:VM)-[:HOSTS]->(:NE)
CREATE (:NE)-[:CONNECTS_TO {bandwidth: $bandwidth}]->(:NE)
```

**2.2 数据批量导入（CSV方式）**

```bash
# 将CSV文件放入容器挂载目录
cp movies.csv /data/import/

# 在Neo4j Browser执行
LOAD CSV WITH HEADERS FROM 'file:///movies.csv' AS row
CREATE (m:Movie { 
    movieId: row.id,
    title: row.title,
    year: toInteger(row.year),
    genre: split(row.genres, '|')
});
```

---

#### **阶段3：Agno智能体系统架构（5小时）**

**3.1 系统架构图**

```
                                +-------------------+
                                | Orchestrator Agent|
                                +--------+----------+
                                         |
                  +------------------------+------------------------+
                  |                        |                        |
        +---------+---------+   +----------+-----------+   +--------+----------+
        |  Input Agents     |   |  Processing Agents   |   |  Output Agents    |
        +---------+---------+   +----------+-----------+   +--------+----------+
                  |                        |                        |
       +----------+----------+  +----------+----------+  +----------+----------+
       | Intent Analyzer     |  | Schema Fetcher      |  | Result Formatter   |
       | Entity Recognizer   |  | Graph Embedding     |  | Visualization      |
       | Query Parser        |  | Vector Retrieval    |  | Cache Manager      |
       | Context Manager     |  | Cypher Generator    |  | Response Synthesizer|
       | Message Router      |  | Syntax Checker      |  | Log Recorder       |
       +---------------------+  | Query Executor      |  +---------------------+
                                | Topology Analyzer   |
                                | Performance Analyzer|
                                | Fault Diagnoser     |
                                +---------------------+
```

**3.2 核心智能体实现**（大模型采用本地部署通过vLLM加载的大模型Qwen2.5-vl-7b）
```python
from agno import Agent, Message, MessageBus

class OrchestratorAgent(Agent):
    """
    总管智能体：负责任务分解、流程编排和智能体协调
    """
    def __init__(self, bus):
        super().__init__("orchestrator")
        self.message_bus = bus
        self.workflow_templates = {
            "basic_query": ["intent_analyzer", "schema_fetcher", "entity_recognizer", 
                          "cypher_generator", "syntax_checker", "query_executor", 
                          "result_formatter"],
            "semantic_query": ["intent_analyzer", "schema_fetcher", "entity_recognizer", 
                             "graph_embedding", "vector_retrieval", "cypher_generator", 
                             "syntax_checker", "query_executor", "result_formatter"],
            "topology_analysis": ["intent_analyzer", "schema_fetcher", "topology_analyzer", 
                                "cypher_generator", "query_executor", "result_formatter"],
            "performance_analysis": ["intent_analyzer", "schema_fetcher", "performance_analyzer", 
                                   "graph_embedding", "vector_retrieval", "cypher_generator", 
                                   "query_executor", "result_formatter"],
            "fault_diagnosis": ["intent_analyzer", "schema_fetcher", "fault_diagnoser", 
                              "graph_embedding", "vector_retrieval", "cypher_generator", 
                              "query_executor", "result_formatter"]
        }
    
    async def on_message(self, msg: Message):
        # 步骤1：意图分析
        intent_msg = await self.request("intent_analyzer", msg)
        intent = intent_msg.data.get("intent", "basic_query")
        
        # 步骤2：获取工作流程
        workflow = self.workflow_templates.get(intent, self.workflow_templates["basic_query"])
        
        # 步骤3：追踪上下文
        msg = msg.update(
            intent=intent,
            workflow=workflow,
            workflow_step=0,
            workflow_results={}
        )
        
        # 步骤4：执行工作流程（串行执行每个智能体）
        current_msg = msg
        for agent_name in workflow:
            try:
                current_msg = await self.request(agent_name, current_msg)
                # 保存每个步骤的结果
                current_msg.workflow_results[agent_name] = current_msg.data
                current_msg.workflow_step += 1
            except Exception as e:
                # 错误处理：记录异常并尝试恢复
                current_msg = current_msg.update(
                    error=f"在执行{agent_name}时出错: {str(e)}",
                    workflow_step=-1  # 标记失败
                )
                break
        
        # 步骤5：返回最终结果
        return current_msg
```

**3.3 意图分析智能体 (适配5G核心网场景)**

```python
class IntentAnalyzerAgent(Agent):
    """
    意图分析智能体：识别用户查询的类型和目标
    """
    def __init__(self, llm_endpoint):
        super().__init__("intent_analyzer")
        self.llm = OpenAI(base_url=llm_endpoint)
        
    async def on_message(self, msg: Message):
        query = msg.data.get("query", "")
        
        # 使用LLM进行意图识别
        prompt = f"""识别以下关于5G核心网资源的查询意图：
        查询：{query}
        
        可能的意图类型：
        1. basic_query - 基础信息查询，如"查找VM_001虚机的详细信息"
        2. semantic_query - 语义查询，如"找到所有性能异常的设备"
        3. topology_analysis - 拓扑分析，如"分析南京数据中心的网络拓扑"
        4. performance_analysis - 性能分析，如"分析过去24小时内CPU使用率超过80%的虚机"
        5. fault_diagnosis - 故障诊断，如"诊断VM_001虚机的故障原因"
        
        返回JSON格式，包含：
        1. intent: 意图类型
        2. target_resources: 目标资源类型列表
        3. conditions: 查询条件
        4. time_range: 时间范围(如有)
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        intent_data = json.loads(response.choices[0].message.content)
        return msg.update(
            intent=intent_data.get("intent", "basic_query"),
            target_resources=intent_data.get("target_resources", []),
            conditions=intent_data.get("conditions", {}),
            time_range=intent_data.get("time_range", None)
        )
```

**3.4 图嵌入智能体 (基于双流架构)**

```python
class GraphEmbeddingAgent(Agent):
    """
    图嵌入智能体：负责生成图数据的向量表示
    """
    def __init__(self, model_config):
        super().__init__("graph_embedding")
        self.model = self._initialize_model(model_config)
        
    def _initialize_model(self, config):
        """初始化图嵌入模型"""
        model_type = config.get("model_type", "dual_stream")
        
        if model_type == "dual_stream":
            # 双流图嵌入模型（静态+动态特征）
            return DualStreamGraphEncoder(
                static_dim=config.get("static_dim", 256),
                dynamic_dim=config.get("dynamic_dim", 128),
                output_dim=config.get("output_dim", 768)
            )
        elif model_type == "text_only":
            # 仅文本嵌入模型
            return SentenceTransformer(config.get("model_name", "all-MiniLM-L6-v2"))
        else:
            # 混合模型
            return HybridGraphEmbedding(config)
    
    async def on_message(self, msg: Message):
        embed_type = msg.data.get("embed_type", "query")
        
        if embed_type == "query":
            # 查询文本嵌入
            query = msg.data.get("query", "")
            vector = self.model.encode_query(query)
        elif embed_type == "node":
            # 节点嵌入
            node_data = msg.data.get("node_data", {})
            vector = self.model.encode_node(node_data)
        elif embed_type == "subgraph":
            # 子图嵌入
            subgraph_data = msg.data.get("subgraph_data", {})
            vector = self.model.encode_subgraph(subgraph_data)
        
        return msg.update(
            embedding_vector=vector,
            embedding_dimension=len(vector)
        )
```

**3.5 拓扑分析智能体 (5G专用)**

```python
class TopologyAnalyzerAgent(Agent):
    """
    拓扑分析智能体：分析5G核心网资源拓扑结构
    """
    def __init__(self, neo4j_uri):
        super().__init__("topology_analyzer")
        self.graph = Graph(neo4j_uri, auth=("neo4j", "agno123"))
        self.analysis_types = {
            "path_analysis": self._analyze_path,
            "connectivity": self._analyze_connectivity,
            "centrality": self._analyze_centrality,
            "community": self._analyze_community
        }
        
    async def on_message(self, msg: Message):
        analysis_type = msg.data.get("analysis_type", "path_analysis")
        target_resources = msg.data.get("target_resources", [])
        
        # 执行对应类型的拓扑分析
        analysis_func = self.analysis_types.get(analysis_type, self._analyze_path)
        analysis_result = analysis_func(target_resources, msg.data)
        
        # 生成Cypher查询
        cypher_query = self._generate_cypher_query(analysis_type, analysis_result)
        
        return msg.update(
            topology_analysis=analysis_result,
            cypher=cypher_query
        )
    
    def _analyze_path(self, resources, data):
        """分析资源之间的路径"""
        # 实现路径分析逻辑
        pass
    
    def _generate_cypher_query(self, analysis_type, result):
        """根据分析结果生成Cypher查询"""
        # 实现Cypher生成逻辑
        pass
```

**3.6 性能分析智能体**

```python
class PerformanceAnalyzerAgent(Agent):
    """
    性能分析智能体：分析5G核心网资源性能数据
    """
    def __init__(self, neo4j_uri):
        super().__init__("performance_analyzer")
        self.graph = Graph(neo4j_uri, auth=("neo4j", "agno123"))
        
    async def on_message(self, msg: Message):
        target_resources = msg.data.get("target_resources", [])
        metrics = msg.data.get("metrics", ["cpu", "memory", "throughput"])
        time_range = msg.data.get("time_range", {"start": "24h", "end": "now"})
        
        # 构造性能查询
        perf_query = self._build_performance_query(target_resources, metrics, time_range)
        
        # 执行查询
        results = self.graph.run(perf_query).data()
        
        # 性能数据分析
        analysis = self._analyze_performance(results, metrics)
        
        # 生成Cypher查询
        cypher_query = self._generate_cypher_query(analysis)
        
        return msg.update(
            performance_analysis=analysis,
            cypher=cypher_query
        )
    
    def _build_performance_query(self, resources, metrics, time_range):
        """构建性能数据查询"""
        pass
    
    def _analyze_performance(self, results, metrics):
        """分析性能数据"""
        pass
```

**3.7 故障诊断智能体**

```python
class FaultDiagnoserAgent(Agent):
    """
    故障诊断智能体：诊断5G核心网故障
    """
    def __init__(self, neo4j_uri, vector_store_client, llm_endpoint):
        super().__init__("fault_diagnoser")
        self.graph = Graph(neo4j_uri, auth=("neo4j", "agno123"))
        self.vector_store = vector_store_client
        self.llm = OpenAI(base_url=llm_endpoint)
        
    async def on_message(self, msg: Message):
        resource_id = msg.data.get("resource_id")
        symptoms = msg.data.get("symptoms", [])
        time_range = msg.data.get("time_range", {"start": "24h", "end": "now"})
        
        # 收集相关数据
        resource_data = self._collect_resource_data(resource_id, time_range)
        
        # 向量检索类似故障
        similar_faults = await self._retrieve_similar_faults(resource_data, symptoms)
        
        # 使用LLM进行故障分析
        diagnosis = await self._analyze_fault(resource_data, symptoms, similar_faults)
        
        # 生成Cypher查询来获取相关资源
        cypher_query = self._generate_diagnostic_query(diagnosis)
        
        return msg.update(
            fault_diagnosis=diagnosis,
            similar_faults=similar_faults,
            cypher=cypher_query
        )
    
    async def _retrieve_similar_faults(self, resource_data, symptoms):
        """检索类似故障"""
        # 构建故障描述
        fault_desc = f"Resource: {resource_data['type']}, Symptoms: {', '.join(symptoms)}"
        
        # 请求图嵌入
        embed_msg = Message(
            data={
                "embed_type": "query",
                "query": fault_desc
            }
        )
        embed_result = await self.request("graph_embedding", embed_msg)
        vector = embed_result.data.get("embedding_vector")
        
        # 向量检索
        similar_faults = self.vector_store.search(
            collection_name="fault_patterns",
            query_vector=vector,
            limit=5
        )
        
        return similar_faults
```

---

#### **阶段4：提示工程优化（针对5G核心网资源）**

**4.1 动态提示模板**

```python
def build_5g_resource_prompt(schema, query, analysis_type):
    """
    构建针对5G资源查询的动态提示
    
    Args:
        schema: 数据库结构信息
        query: 用户查询
        analysis_type: 分析类型
        
    Returns:
        构建好的提示
    """
    base_prompt = f"""## 5G核心网资源查询任务
### 数据库结构
{format_schema(schema)}

### 用户原始问题
{query}
"""
    
    if analysis_type == "basic_query":
        return base_prompt + """
### 生成规则
1. 使用精确的标签和属性匹配
2. 优先使用索引字段(id, name)
3. 返回结果限制为20条
4. 包含必要的属性筛选
"""
    elif analysis_type == "topology_analysis":
        return base_prompt + """
### 拓扑分析规则
1. 使用图算法函数 (如 gds.* 函数)
2. 分析网元之间的连接关系
3. 找出关键路径和节点
4. 按照重要性排序结果
"""
    elif analysis_type == "performance_analysis":
        return base_prompt + """
### 性能分析规则
1. 包含时间范围过滤 (使用 timestamp() 函数)
2. 聚合性能指标 (使用 avg(), max(), min())
3. 设置阈值条件 (例如 CPU > 80%)
4. 按性能指标排序结果
"""
    else:  # fault_diagnosis
        return base_prompt + """
### 故障诊断规则
1. 查找相关联的告警信息
2. 分析资源依赖关系
3. 包含状态异常的条件
4. 返回可能的根因和影响范围
"""
```

**4.2 多轮对话与上下文管理**

```python
class ContextManagerAgent(Agent):
    """
    上下文管理智能体：维护对话历史和上下文信息
    """
    def __init__(self):
        super().__init__("context_manager")
        self.sessions = {}  # {session_id: [history]}
        self.max_history = 5  # 保留最近5轮对话
        
    async def on_message(self, msg: Message):
        session_id = msg.meta.get('session_id', 'default')
        
        # 初始化会话
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        # 获取当前查询
        query = msg.data.get('query', '')
        
        # 如果是澄清或补充
        if msg.data.get('intent') == 'clarify':
            # 获取上一轮对话
            if self.sessions[session_id]:
                last_context = self.sessions[session_id][-1]
                
                # 合并上下文
                context = self._merge_context(last_context, {
                    'query': query,
                    'timestamp': time.time()
                })
                
                # 更新消息
                msg = msg.update(
                    original_query=last_context.get('query'),
                    clarification=query,
                    context=context
                )
        else:
            # 保存当前查询到历史
            self.sessions[session_id].append({
                'query': query,
                'timestamp': time.time(),
                'intent': msg.data.get('intent'),
                'results': None  # 稍后会更新
            })
            
            # 保持历史长度
            if len(self.sessions[session_id]) > self.max_history:
                self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
                
            # 添加历史上下文
            msg = msg.update(
                context=self.sessions[session_id],
                is_followup=len(self.sessions[session_id]) > 1
            )
            
        return msg
        
    def _merge_context(self, prev_context, new_context):
        """合并上下文"""
        merged = prev_context.copy()
        merged.update(new_context)
        return merged
```

---

#### **阶段5：系统集成与API开发**

**5.1 消息总线配置**

```python
from agno import MessageBus

def configure_message_bus(config):
    """配置消息总线和智能体"""
    
    # 创建消息总线
    bus = MessageBus()
    
    # 创建智能体
    orchestrator = OrchestratorAgent(bus)
    intent_analyzer = IntentAnalyzerAgent(config['llm_endpoint'])
    schema_fetcher = SchemaFetcherAgent(config['neo4j_uri'])
    entity_recognizer = EntityRecognizerAgent(config['llm_endpoint'])
    graph_embedding = GraphEmbeddingAgent(config['embedding_config'])
    vector_retrieval = VectorRetrievalAgent(config['vector_db_config'])
    cypher_generator = CypherGeneratorAgent(config['llm_endpoint'])
    syntax_checker = SyntaxCheckerAgent(config['neo4j_uri'])
    query_executor = QueryExecutorAgent(config['neo4j_uri'])
    result_formatter = ResultFormatterAgent(config['llm_endpoint'])
    
    # 5G专用智能体
    topology_analyzer = TopologyAnalyzerAgent(config['neo4j_uri'])
    performance_analyzer = PerformanceAnalyzerAgent(config['neo4j_uri'])
    fault_diagnoser = FaultDiagnoserAgent(
        config['neo4j_uri'],
        config['vector_db_client'],
        config['llm_endpoint']
    )
    
    # 注册智能体
    bus.register_agents([
        orchestrator,
        intent_analyzer,
        schema_fetcher,
        entity_recognizer,
        graph_embedding,
        vector_retrieval,
        cypher_generator,
        syntax_checker,
        query_executor,
        result_formatter,
        topology_analyzer,
        performance_analyzer,
        fault_diagnoser
    ])
    
    return bus
```

**5.2 FastAPI集成**

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import json

# 创建API应用
app = FastAPI(
    title="5G核心网资源图谱查询API",
    description="基于Agno框架的多智能体自然语言转Cypher查询系统",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载配置
with open("config.json") as f:
    config = json.load(f)

# 配置消息总线
bus = configure_message_bus(config)

# API模型
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    analysis_type: str = None
    include_vectors: bool = False
    top_k: int = 5

class QueryResponse(BaseModel):
    cypher: str
    results: list
    analysis: dict = None
    execution_time: float
    workflow_trace: dict = None

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """处理自然语言查询并返回结果"""
    start_time = time.time()
    
    # 初始化消息
    msg = Message(
        data={
            "query": request.query,
            "analysis_type": request.analysis_type,
            "top_k": request.top_k,
            "include_vectors": request.include_vectors
        },
        meta={
            "session_id": request.session_id,
            "timestamp": start_time
        }
    )
    
    # 通过总管智能体处理请求
    result = await bus.process("orchestrator", msg)
    
    if result.error:
        raise HTTPException(status_code=400, detail=result.error)
    
    # 准备响应
    response = {
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
    if config.get("debug_mode", False):
        response["workflow_trace"] = result.workflow_results
    
    return response

@app.get("/api/v1/schema")
async def get_schema():
    """获取数据库模式"""
    schema_msg = Message()
    result = await bus.process("schema_fetcher", schema_msg)
    return result.data.get("schema", {})
```

---

#### **阶段6：测试与生产部署**

**6.1 5G资源图谱测试用例**

```python
test_cases = [
    {
        "name": "基础资源查询",
        "input": "查找名为VM_001的虚拟机及其部署的网元",
        "expected_intent": "basic_query",
        "expected_cypher": "MATCH (v:VM {name:'VM_001'})-[:HOSTS]->(n:NE) RETURN v.name AS vm_name, v.status AS vm_status, n.name AS ne_name, n.type AS ne_type LIMIT 20"
    },
    {
        "name": "性能分析查询",
        "input": "查找过去24小时内CPU使用率超过90%的所有虚拟机",
        "expected_intent": "performance_analysis",
        "expected_cypher": "MATCH (v:VM) WHERE v.cpu > 90 AND v.timestamp >= datetime() - duration('PT24H') RETURN v.name AS name, v.cpu AS cpu_usage, v.memory AS memory_usage ORDER BY v.cpu DESC LIMIT 20"
    },
    {
        "name": "拓扑分析查询",
        "input": "分析南京数据中心的网络拓扑结构",
        "expected_intent": "topology_analysis"
    },
    {
        "name": "故障诊断查询",
        "input": "分析VM_042虚拟机的故障原因及影响范围",
        "expected_intent": "fault_diagnosis"
    }
]

async def run_tests(bus):
    """运行测试用例"""
    results = []
    
    for case in test_cases:
        # 创建消息
        msg = Message(
            data={"query": case["input"]},
            meta={"session_id": f"test-{uuid.uuid4()}"}
        )
        
        # 处理消息
        result = await bus.process("orchestrator", msg)
        
        # 验证结果
        test_result = {
            "name": case["name"],
            "input": case["input"],
            "expected_intent": case["expected_intent"],
            "actual_intent": result.data.get("intent"),
            "cypher": result.data.get("cypher"),
            "passed": result.data.get("intent") == case["expected_intent"]
        }
        
        # 检查Cypher（如果有预期）
        if "expected_cypher" in case:
            cypher_similar = compare_cypher(
                case["expected_cypher"],
                result.data.get("cypher", "")
            )
            test_result["cypher_match"] = cypher_similar
            test_result["passed"] = test_result["passed"] and cypher_similar
        
        results.append(test_result)
    
    return results

def compare_cypher(expected, actual):
    """
    比较预期与实际的Cypher查询
    使用基于抽象语法树的比较而非纯文本匹配
    """
    try:
        # 解析查询至AST表示
        expected_ast = parse_cypher(expected)
        actual_ast = parse_cypher(actual)
        
        # 比较关键组件
        similarity_score = compare_ast_nodes(expected_ast, actual_ast)
        return similarity_score > 0.8  # 80%相似度阈值
    except:
        # 回退到简单的文本相似度
        return simple_text_similarity(expected, actual) > 0.7
```

**6.2 系统监控与日志记录**

```python
def setup_monitoring(app, config):
    """
    设置系统监控和日志记录
    """
    # 设置结构化日志
    import logging
    import json_log_formatter

    formatter = json_log_formatter.JSONFormatter()
    json_handler = logging.FileHandler(filename=config['log_path'])
    json_handler.setFormatter(formatter)
    
    logger = logging.getLogger('cypher_agent')
    logger.addHandler(json_handler)
    logger.setLevel(logging.INFO)
    
    # 记录重要事件
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 记录请求信息
        logger.info(
            "request_processed",
            extra={
                "request_path": request.url.path,
                "request_method": request.method,
                "status_code": response.status_code,
                "process_time": process_time,
            }
        )
        
        return response
    
    # 性能指标导出（Prometheus格式）
    if config.get('enable_metrics', False):
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app)
```

**6.3 部署配置**

```yaml
# docker-compose.yml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.12
    container_name: neo4j_agno
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - ./import:/var/lib/neo4j/import
    environment:
      - NEO4J_AUTH=neo4j/agno123
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
      - NEO4J_dbms_memory_heap_initial__size=2G
      - NEO4J_dbms_memory_heap_max__size=4G
    restart: unless-stopped
    
  vector_db:
    image: milvusdb/milvus:v2.3.3
    container_name: milvus_agno
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - milvus_data:/var/lib/milvus
    environment:
      - MILVUS_HOST=0.0.0.0
      - MILVUS_PORT=19530
    restart: unless-stopped
    
  cypher_agent:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: cypher_agent
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - MILVUS_HOST=vector_db
      - MILVUS_PORT=19530
      - MODEL_ENDPOINT=http://llm_service:8080/v1
      - LOG_LEVEL=INFO
    depends_on:
      - neo4j
      - vector_db
      - llm_service
    restart: unless-stopped
    
  llm_service:
    image: ghcr.io/huggingface/text-generation-inference:latest
    container_name: llm_service
    ports:
      - "8080:80"
    volumes:
      - ./models:/models
    environment:
      - MODEL_ID=/models/qwen2.5-7b
      - MAX_INPUT_LENGTH=4096
      - MAX_TOTAL_TOKENS=8192
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  neo4j_data:
  milvus_data:
```

---

## 阶段7：双流图嵌入模型集成

**7.1 图嵌入模型架构**

```python
class DualStreamGraphEncoder(nn.Module):
    """双流图嵌入模型：结合静态和动态特征"""
    
    def __init__(
        self,
        static_dim=256,
        dynamic_dim=128,
        hidden_dim=512,
        output_dim=768,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        # 静态特征编码器
        self.static_encoder = StaticGraphEncoder(
            node_dim=static_dim,
            edge_dim=static_dim // 2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 动态特征编码器
        self.dynamic_encoder = DynamicGraphEncoder(
            metrics_dim=dynamic_dim,
            seq_len=24,  # 时间窗口长度
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 特征融合层
        self.fusion = CrossStreamFusion(
            static_dim=hidden_dim,
            dynamic_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
    def forward(
        self,
        node_features,
        edge_index,
        edge_features,
        time_series=None,
        batch=None
    ):
        # 编码静态特征
        static_embeddings = self.static_encoder(
            node_features,
            edge_index,
            edge_features
        )
        
        # 编码动态特征（如果有）
        if time_series is not None:
            dynamic_embeddings = self.dynamic_encoder(time_series)
        else:
            # 如果没有时间序列数据，使用零向量
            dynamic_embeddings = torch.zeros_like(static_embeddings)
        
        # 融合特征
        output_embeddings = self.fusion(
            static_embeddings,
            dynamic_embeddings
        )
        
        return output_embeddings
```

**7.2 模型接口与查询系统集成**

```python
class GraphEmbeddingService:
    """
    图嵌入服务：提供向量化和检索功能
    """
    def __init__(self, model_path, vector_db_config):
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 初始化向量数据库客户端
        self.vector_db = self._init_vector_db(vector_db_config)
        
    def _load_model(self, model_path):
        """加载预训练模型"""
        if os.path.exists(model_path):
            model = DualStreamGraphEncoder()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    def _init_vector_db(self, config):
        """初始化向量数据库连接"""
        if config['type'] == 'milvus':
            from pymilvus import connections, Collection
            connections.connect(
                alias="default",
                host=config['host'],
                port=config['port']
            )
            return {
                'nodes': Collection('node_embeddings'),
                'paths': Collection('path_embeddings'),
                'faults': Collection('fault_embeddings')
            }
        else:
            # 使用内存向量存储的简单实现
            return InMemoryVectorStore()
    
    def encode_query(self, query_text):
        """将查询文本编码为向量"""
        # 使用文本编码器
        return self.model.encode_text(query_text)
    
    def encode_node(self, node_data):
        """将节点数据编码为向量"""
        # 准备节点特征
        node_features = self._prepare_node_features(node_data)
        
        # 使用模型编码
        with torch.no_grad():
            embedding = self.model.encode_node(node_features)
            
        return embedding.cpu().numpy()
    
    def search_similar_nodes(self, query_vector, top_k=5):
        """搜索相似节点"""
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.vector_db['nodes'].search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["node_id", "node_type", "properties"]
        )
        
        return self._format_search_results(results)
```

**7.3 整合到Agno框架**

```python
# 修改GraphEmbeddingAgent以支持双流模型
class GraphEmbeddingAgent(Agent):
    """
    图嵌入智能体：负责生成图数据的向量表示
    """
    def __init__(self, model_config):
        super().__init__("graph_embedding")
        self.model = self._initialize_model(model_config)
        
    def _initialize_model(self, config):
        """初始化图嵌入模型"""
        model_type = config.get("model_type", "dual_stream")
        
        if model_type == "dual_stream":
            # 加载双流图嵌入模型
            model_path = config.get("model_path", "models/dual_stream_encoder.pt")
            return GraphEmbeddingService(
                model_path=model_path,
                vector_db_config=config.get("vector_db", {})
            )
        else:
            # 回退到基本文本编码器
            return TextEmbeddingService(config)
    
    async def on_message(self, msg: Message):
        embed_type = msg.data.get("embed_type", "query")
        
        try:
            if embed_type == "query":
                # 查询文本嵌入
                query = msg.data.get("query", "")
                vector = self.model.encode_query(query)
            elif embed_type == "node":
                # 节点嵌入
                node_data = msg.data.get("node_data", {})
                vector = self.model.encode_node(node_data)
            elif embed_type == "subgraph":
                # 子图嵌入
                subgraph_data = msg.data.get("subgraph_data", {})
                vector = self.model.encode_subgraph(subgraph_data)
            
            return msg.update(
                embedding_vector=vector,
                embedding_dimension=len(vector),
                error=None
            )
        except Exception as e:
            return msg.update(
                error=f"图嵌入生成失败: {str(e)}"
            )
```

---

## 阶段8：系统实际应用案例

**8.1 拓扑分析应用案例**

```python
# 初始化系统
system = configure_message_bus({
    'neo4j_uri': 'bolt://localhost:7687',
    'llm_endpoint': 'http://localhost:8080/v1',
    'embedding_config': {
        'model_type': 'dual_stream',
        'model_path': 'models/dual_stream_encoder.pt'
    },
    'vector_db_config': {
        'type': 'milvus',
        'host': 'localhost',
        'port': 19530
    }
})

# 拓扑分析查询
async def topology_analysis_example():
    query = "分析南京数据中心的网络拓扑，找出关键连接点和可能的单点故障"
    
    # 创建消息
    msg = Message(
        data={"query": query},
        meta={"session_id": "demo-session"}
    )
    
    # 通过总管智能体处理请求
    result = await system.process("orchestrator", msg)
    
    # 输出结果
    print(f"生成的Cypher查询: {result.data.get('cypher')}")
    print(f"关键节点: {result.data.get('topology_analysis', {}).get('critical_nodes', [])}")
    print(f"单点故障风险: {result.data.get('topology_analysis', {}).get('single_point_failures', [])}")
```

**8.2 故障诊断应用案例**

```python
# 故障诊断查询
async def fault_diagnosis_example():
    query = "VM_042最近2小时内出现CPU使用率异常，分析可能的原因和影响范围"
    
    # 创建消息
    msg = Message(
        data={"query": query},
        meta={"session_id": "demo-session"}
    )
    
    # 通过总管智能体处理请求
    result = await system.process("orchestrator", msg)
    
    # 输出结果
    print(f"生成的Cypher查询: {result.data.get('cypher')}")
    print(f"故障诊断: {result.data.get('fault_diagnosis', {}).get('root_cause')}")
    print(f"影响范围: {result.data.get('fault_diagnosis', {}).get('impact_assessment')}")
    print(f"推荐操作: {result.data.get('fault_diagnosis', {}).get('recommendations')}")
```

**8.3 性能分析应用案例**

```python
# 性能分析查询
async def performance_analysis_example():
    query = "分析南京数据中心最近7天的性能趋势，识别潜在的性能瓶颈"
    
    # 创建消息
    msg = Message(
        data={"query": query},
        meta={"session_id": "demo-session"}
    )
    
    # 通过总管智能体处理请求
    result = await system.process("orchestrator", msg)
    
    # 输出结果
    print(f"生成的Cypher查询: {result.data.get('cypher')}")
    
    # 获取性能分析结果
    perf_analysis = result.data.get('performance_analysis', {})
    
    # 输出瓶颈点
    print(f"性能瓶颈: {perf_analysis.get('bottlenecks', [])}")
    
    # 输出趋势分析
    print(f"性能趋势: {perf_analysis.get('trends', {})}")
    
    # 输出容量预测
    print(f"容量预测: {perf_analysis.get('capacity_forecast', {})}")
```

**8.4 资源关联关系查询**

```python
# 资源关联关系查询
async def resource_relationship_example():
    query = "查找与核心网网元NE_105有依赖关系的所有虚拟机，以及这些虚拟机所在的物理服务器"
    
    # 创建消息
    msg = Message(
        data={"query": query},
        meta={"session_id": "demo-session"}
    )
    
    # 通过总管智能体处理请求
    result = await system.process("orchestrator", msg)
    
    # 输出结果
    print(f"生成的Cypher查询: {result.data.get('cypher')}")
    print(f"依赖关系: {result.data.get('results')}")
```

---

## 阶段9：未来扩展计划

**9.1 多模态交互能力**
- 支持图形化查询界面
- 整合图表可视化功能
- 添加自然语言响应解释

**9.2 智能推荐与预测**
- 资源使用预测
- 故障风险预警
- 容量规划建议

**9.3 持续学习能力**
- 从历史查询中改进
- 适应新的资源类型和关系
- 更新故障模式库

**9.4 高级安全特性**
- 查询权限控制
- 敏感数据过滤
- 审计跟踪功能

---

本设计文档集成了多种先进技术，包括基于Agno框架的多智能体协同系统、双流图嵌入模型、向量检索和大语言模型，为5G核心网资源图谱查询提供了全面的解决方案。通过提供灵活的架构和丰富的功能，系统能够满足从基础查询到复杂分析的各种需求，助力网络运维和故障诊断工作。

