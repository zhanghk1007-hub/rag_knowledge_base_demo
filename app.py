"""
RAG (检索增强生成) 知识库问答系统
================================================
基于 Streamlit + Ollama + Milvus Lite 的本地知识库问答应用

作者: 张淮宽
用途: 展示 RAG 系统的完整工作流程
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd

# =============================================================================
# 配置区域 - 可以根据需要修改这些常量
# =============================================================================

# 方法：直接存在当前脚本同级目录下，自动兼容所有电脑
import os

MILVUS_DB_PATH = "./milvus_demo.db" 


# Ollama 模型配置
EMBEDDING_MODEL = "mxbai-embed-large"  # 嵌入模型，用于将文本转为向量
CHAT_MODEL = "gemma3:1b"                       # 聊天模型，用于生成回答
VECTOR_DIM = 1024                              # 向量维度，与嵌入模型输出一致

# 文本分段配置
CHUNK_SIZE = 150      # 每个文本片段的最大字符数
CHUNK_OVERLAP = 10    # 片段之间的重叠字符数（用于保持上下文连贯性）

# Milvus Collection 名称
COLLECTION_NAME = "rag_demo"

# =============================================================================
# 页面配置 - 设置 Streamlit 页面的标题和布局
# =============================================================================

st.set_page_config(
    page_title="📚 RAG 知识库问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 初始化函数 - 应用启动时执行的检测和初始化
# =============================================================================

@st.cache_resource
def initialize_system() -> Dict[str, Any]:
    """
    系统初始化函数
    
    这个函数在应用启动时自动执行，完成以下任务：
    1. 检测并创建 Milvus 数据库目录
    2. 连接到 Milvus Lite 数据库
    3. 检查并创建必要的 Collection
    4. 检测 Ollama 服务是否可用
    
    Returns:
        Dict 包含初始化状态和错误信息
    """
    status = {
        "milvus_ready": False,
        "ollama_ready": False,
        "collection_ready": False,
        "errors": []
    }
    
    # -------------------- 步骤 1: 初始化 Milvus 数据库目录 --------------------
    try:
        # 获取数据库文件的目录路径
        db_dir = os.path.dirname(MILVUS_DB_PATH)
        
        # 如果目录不存在，尝试创建它
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"✅ 创建数据库目录: {db_dir}")
        
        status["milvus_ready"] = True
    except Exception as e:
        status["errors"].append(f"❌ 数据库目录初始化失败: {str(e)}")
    
    # -------------------- 步骤 2: 连接 Milvus 并检查 Collection --------------------
    if status["milvus_ready"]:
        try:
            from pymilvus import MilvusClient, DataType
            
            # 创建 Milvus Lite 客户端连接
            # Milvus Lite 会在指定路径自动创建数据库文件
            client = MilvusClient(uri=MILVUS_DB_PATH)
            
            # 检查 Collection 是否存在
            collections = client.list_collections()
            
            if COLLECTION_NAME not in collections:
                # 如果 Collection 不存在，创建一个新的
                # 定义 Schema（数据结构）
                schema = MilvusClient.create_schema(
                    auto_id=True,  # 自动生成主键 ID
                    enable_dynamic_field=True
                )
                
                # 添加字段到 Schema
                # 1. 主键 ID 字段（自动生成）
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                
                # 2. 向量字段 - 存储 1024 维的文本嵌入向量
                schema.add_field(
                    field_name="vector", 
                    datatype=DataType.FLOAT_VECTOR, 
                    dim=VECTOR_DIM
                )
                
                # 3. 来源文件字段 - 记录这段文本来自哪个文件
                schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=500)
                
                # 4. 文本内容字段 - 存储原始文本片段
                schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
                
                # 5. 日期字段 - 记录上传时间
                schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=50)
                
                # 创建 Collection
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    schema=schema
                )
                
                # 创建向量索引 - 用于加速相似度搜索
                # IVF_FLAT 是一种常用的近似最近邻搜索索引
                index_params = MilvusClient.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_type="IVF_FLAT",  # 索引类型
                    metric_type="COSINE",   # 相似度度量方式：余弦相似度
                    params={"nlist": 128}   # 聚类中心数量
                )
                client.create_index(
                    collection_name=COLLECTION_NAME,
                    index_params=index_params
                )
                
                print(f"✅ 创建 Collection: {COLLECTION_NAME}")
            else:
                print(f"✅ Collection 已存在: {COLLECTION_NAME}")
            
            # 加载 Collection 到内存（Milvus 要求搜索前必须先加载）
            client.load_collection(COLLECTION_NAME)
            status["collection_ready"] = True
            
        except Exception as e:
            status["errors"].append(f"❌ Milvus 初始化失败: {str(e)}")
    
    # -------------------- 步骤 3: 检测 Ollama 服务 --------------------
    try:
        import ollama
        # 尝试列出本地模型，验证 Ollama 服务是否运行
        ollama.list()
        status["ollama_ready"] = True
        print("✅ Ollama 服务检测正常")
    except Exception as e:
        status["errors"].append(
            f"❌ Ollama 服务未启动或无法连接。请确保:\n"
            f"   1. Ollama 已安装: https://ollama.com\n"
            f"   2. Ollama 服务正在运行: 在终端执行 `ollama serve`\n"
            f"   3. 已下载所需模型: `ollama pull {EMBEDDING_MODEL}` 和 `ollama pull {CHAT_MODEL}`"
        )
    
    return status

# =============================================================================
# 文本处理函数 - 文件解析和分段
# =============================================================================

def parse_file(uploaded_file) -> str:
    """
    解析上传的文件内容
    
    Args:
        uploaded_file: Streamlit 上传的文件对象
    
    Returns:
        文件的文本内容
    """
    # 读取文件内容为字节
    bytes_data = uploaded_file.getvalue()
    
    # 尝试用 UTF-8 解码，如果失败则使用 GBK（中文 Windows 常用编码）
    try:
        text = bytes_data.decode('utf-8')
    except UnicodeDecodeError:
        text = bytes_data.decode('gbk', errors='ignore')
    
    return text

# =============================================================================
# [修改] 升级后的文本分段函数
# =============================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, separator: str = None) -> List[str]:
    """
    升级版分段函数：支持自定义分隔符
    """
    final_chunks = []
    
    # 1. 如果没有指定分隔符，直接使用原来的按长度滑动窗口切分
    if not separator:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunk = chunk.lstrip('\n')
            if chunk:
                final_chunks.append(chunk)
            start += (chunk_size - overlap)
        return final_chunks

    # 2. 如果指定了分隔符，先按分隔符粗切
    # 处理用户输入的转义字符，例如把 "\n" 转为真正的换行符
    real_separator = separator.replace("\\n", "\n").replace("\\t", "\t")
    
    # 按分隔符切分
    raw_pieces = text.split(real_separator)
    
    for piece in raw_pieces:
        piece = piece.strip()
        if not piece:
            continue
            
        # 3. 检查切分后的片段是否依然过长
        if len(piece) > chunk_size:
            # 如果某一段依然太长（超过了嵌入模型的限制），递归调用自己进行强制切分
            # 注意：这里 separator=None，强制进入上面的“按长度切分”逻辑
            sub_chunks = chunk_text(piece, chunk_size, overlap, separator=None)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(piece)
            
    return final_chunks


# =============================================================================
# 嵌入和向量操作函数
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    调用 Ollama 嵌入模型将文本转换为向量
    
    原理说明：
    - 嵌入（Embedding）是将人类语言转换为计算机能理解的数字表示
    - 语义相似的文本在向量空间中距离较近
    - 1024 维向量意味着每个文本被表示为 1024 个浮点数
    
    Args:
        text: 输入文本
    
    Returns:
        1024 维的浮点数向量
    """
    import ollama
    
    # 调用 Ollama 的嵌入 API
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )
    
    return response['embedding']

def search_similar(query: str, top_k: int = 3) -> List[Dict]:
    """
    在 Milvus 中搜索与查询最相似的文本片段
    
    这是 RAG 的核心步骤：
    1. 将用户问题转换为向量
    2. 在向量数据库中查找最相似的文本片段
    3. 返回最相关的上下文用于生成回答
    
    Args:
        query: 用户查询文本
        top_k: 返回最相似的 k 个结果
    
    Returns:
        相似文本片段列表，每个包含 source, text, distance 等信息
    """
    from pymilvus import MilvusClient
    
    # 连接 Milvus
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    # 将查询文本转为向量
    query_vector = get_embedding(query)
    
    # 执行相似度搜索
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],  # 搜索向量（可以是多个）
        limit=top_k,          # 返回最相似的 top_k 个结果
        output_fields=["source", "text", "date"]  # 需要返回的字段
    )
    
    # 格式化搜索结果
    hits = []
    if results and len(results) > 0:
        for hit in results[0]:  # 取第一个查询向量的结果
            hits.append({
                "id": hit.get("id", "N/A"),
                "source": hit.get("entity", {}).get("source", "未知"),
                "text": hit.get("entity", {}).get("text", ""),
                "date": hit.get("entity", {}).get("date", ""),
                "distance": hit.get("distance", 0),  # 相似度分数
            })
    
    return hits

# =============================================================================
# 大模型生成函数
# =============================================================================

def generate_answer(query: str, contexts: List[Dict]) -> str:
    """
    调用 Ollama 大模型生成回答
    
    使用检索到的上下文构建 Prompt，让模型基于知识库内容回答
    
    Args:
        query: 用户问题
        contexts: 检索到的相关文本片段
    
    Returns:
        模型生成的回答
    """
    import ollama
    
    # 构建上下文字符串
    context_text = "\n\n---\n\n".join([
        f"【片段 {i+1}】来源: {ctx['source']}\n内容: {ctx['text']}"
        for i, ctx in enumerate(contexts)
    ])
    
    # 构建系统 Prompt - 指导模型如何回答
    system_prompt = """你是一个基于知识库的问答助手。请根据下面提供的参考资料回答用户的问题。

重要规则：
1. 只使用提供的参考资料回答问题，不要添加外部知识
2. 如果参考资料中没有相关信息，请明确说明"根据现有知识库，我无法回答这个问题"
3. 回答要简洁、准确、有帮助
4. 如果可能，请引用参考资料中的具体内容

参考资料：
"""
    
    # 完整的对话消息
    messages = [
        {
            "role": "system", 
            "content": system_prompt + context_text
        },
        {
            "role": "user", 
            "content": query
        }
    ]
    
    # 调用 Ollama 生成回答
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        options={
            "temperature": 0.7,  # 创造性程度，0-1 之间
            "num_predict": 1024  # 最大生成 token 数
        }
    )
    
    return response['message']['content']

# =============================================================================
# 数据存储函数
# =============================================================================

def store_chunks(chunks: List[str], source_name: str) -> int:
    """
    将文本片段存储到 Milvus 向量数据库
    
    流程：
    1. 对每个文本片段生成嵌入向量
    2. 将向量、原文、来源等信息存入 Milvus
    
    Args:
        chunks: 文本片段列表
        source_name: 来源文件名
    
    Returns:
        成功存储的片段数量
    """
    from pymilvus import MilvusClient
    
    # 连接 Milvus
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    # 准备数据
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 为每个片段生成嵌入并构建数据记录
    data = []
    for chunk in chunks:
        # 生成嵌入向量
        embedding = get_embedding(chunk)
        
        # 构建数据记录
        data.append({
            "vector": embedding,
            "source": source_name,
            "text": chunk,
            "date": current_date
        })
    
    # 批量插入数据
    if data:
        client.insert(
            collection_name=COLLECTION_NAME,
            data=data
        )
    
    return len(data)

# =============================================================================
# [修改] 修复后的清空知识库函数
# =============================================================================

def reset_knowledge_base() -> bool:
    """删除 Collection 并重建，彻底清空数据"""
    from pymilvus import MilvusClient
    try:
        client = MilvusClient(uri=MILVUS_DB_PATH)
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            print(f"🗑️ 已删除 Collection: {COLLECTION_NAME}")
        
        # ---------------------------------------------------------
        # [关键修复] 清除 initialize_system 的缓存
        # 这样下一次 st.rerun() 时，initialize_system 会真正执行，
        # 从而重新创建刚才被删掉的 Collection
        # ---------------------------------------------------------
        initialize_system.clear()
        
        return True
    except Exception as e:
        st.error(f"清空失败: {str(e)}")
        return False



def get_all_documents() -> pd.DataFrame:
    """
    获取 Milvus 中存储的所有文档信息（用于数据透视）
    
    Returns:
        DataFrame 包含所有存储的文本片段信息
    """
    from pymilvus import MilvusClient
    
    # 连接 Milvus
    client = MilvusClient(uri=MILVUS_DB_PATH)
    
    # 查询所有数据（最多返回 10000 条）
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="",  # 空过滤器表示查询所有
        output_fields=["id", "source", "text", "date"],
        limit=10000
    )
    
    # 转换为 DataFrame
    if results:
        df = pd.DataFrame(results)
        # 添加字符数列
        df['字符数'] = df['text'].apply(len)
        # 重命名列
        df = df.rename(columns={
            'id': 'ID',
            'source': '来源文件',
            'text': '文本片段',
            'date': '上传时间'
        })
        # 只保留需要的列
        df = df[['ID', '来源文件', '文本片段', '字符数', '上传时间']]
        return df
    else:
        return pd.DataFrame(columns=['ID', '来源文件', '文本片段', '字符数', '上传时间'])

def get_collection_stats() -> Dict:
    """
    获取 Collection 的统计信息
    
    Returns:
        包含文档数量等统计信息的字典
    """
    from pymilvus import MilvusClient
    
    try:
        client = MilvusClient(uri=MILVUS_DB_PATH)
        stats = client.get_collection_stats(COLLECTION_NAME)
        return stats
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# UI 组件函数
# =============================================================================

# =============================================================================
# [修改] 升级后的侧边栏渲染函数
# =============================================================================

def render_sidebar():
    """
    渲染侧边栏 - 文件上传和处理流程
    """
    st.sidebar.title("📁 知识库管理")
    
    # --- 新增功能区：清空知识库 ---
    with st.sidebar.expander("🗑️ 数据管理", expanded=False):
        st.caption("如果知识库混乱或报错，可以点击下方按钮清空所有数据。")
        if st.button("⚠️ 清空所有知识库", type="secondary", use_container_width=True):
            if reset_knowledge_base():
                st.toast("✅ 知识库已清空，正在重置...", icon="🗑️")
                time.sleep(1)
                st.rerun() # 强制刷新页面以重新初始化
    
    st.sidebar.markdown("---")
    
    # 文件上传组件
    st.sidebar.subheader("1️⃣ 上传文档")
    
    # --- 新增功能区：自定义分隔符 ---
    custom_separator = st.sidebar.text_input(
        "自定义分隔符 (可选)",
        placeholder="例如: \\n\\n 或 ###",
        help="如果填写，系统将优先按此符号切分文本。如果单段过长，系统会自动再次切分。"
    )
    
    uploaded_file = st.sidebar.file_uploader(
        label="选择文件（支持 .md 和 .txt）",
        type=['md', 'txt'],
        help="上传 Markdown 或纯文本文件到知识库"
    )
    
    st.sidebar.caption(
        "💡 原理：将文档内容存入向量数据库，"
        "供后续问答时检索使用"
    )
    
    # 如果用户上传了文件，处理它
    if uploaded_file is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("2️⃣ 处理流程")
        
        # 使用 session_state 避免重复处理同一个文件
        if 'last_uploaded' not in st.session_state:
            st.session_state.last_uploaded = None
        
        # 这里的判断逻辑稍微放宽，允许用户反复点击处理（只要文件名变了或者用户想重试）
        if st.session_state.last_uploaded != uploaded_file.name:
            # 显示处理状态
            with st.sidebar.status("正在处理文档...", expanded=True) as status:
                # -------------------- 步骤 1: 解析 --------------------\n                st.write("📖 **步骤 1/4: 解析文件**")
                text_content = parse_file(uploaded_file)
                st.write(f"   ✅ 成功读取 {len(text_content)} 个字符")
                time.sleep(0.3)
                
                # -------------------- 步骤 2: 分段 --------------------\n                st.write("✂️ **步骤 2/4: 文本分段**")
                
                # [关键修改] 调用新的分段逻辑，传入 custom_separator
                # 如果用户没填，custom_separator 是空字符串，传 None 给函数
                sep_arg = custom_separator if custom_separator.strip() else None
                
                if sep_arg:
                    st.caption(f"正在使用自定义分隔符 `{sep_arg}` 进行切分...")
                else:
                    st.caption(f"按 {CHUNK_SIZE} 字符固定长度切分...")
                
                chunks = chunk_text(text_content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, separator=sep_arg)
                
                st.write(f"   ✅ 生成 {len(chunks)} 个文本片段")
                time.sleep(0.3)
                
                # -------------------- 步骤 3: 嵌入 --------------------\n                st.write("🔢 **步骤 3/4: 生成向量嵌入**")
                progress_bar = st.progress(0)
                for i in range(min(5, len(chunks))):
                    progress_bar.progress((i + 1) / min(5, len(chunks)))
                    time.sleep(0.05)
                progress_bar.empty()
                st.write(f"   ✅ 为 {len(chunks)} 个片段生成嵌入向量")
                
                # -------------------- 步骤 4: 存储 --------------------\n                st.write("💾 **步骤 4/4: 存入向量数据库**")
                
                try:
                    count = store_chunks(chunks, uploaded_file.name)
                    st.write(f"   ✅ 成功存储 {count} 条记录")
                    status.update(label="✅ 文档处理完成！", state="complete", expanded=False)
                    
                    # 记录已处理的文件
                    st.session_state.last_uploaded = uploaded_file.name
                    st.sidebar.success(f"🎉 成功导入 '{uploaded_file.name}'")
                    
                except Exception as e:
                    status.update(label="❌ 处理失败", state="error")
                    st.sidebar.error(f"存储失败: {str(e)}")
        else:
            st.sidebar.info(f"📋 '{uploaded_file.name}' 已处理过")
            if st.sidebar.button("🔄 强制重新处理"):
                st.session_state.last_uploaded = None
                st.rerun()
    
    # 显示知识库统计
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 知识库统计")
    try:
        stats = get_collection_stats()
        # 兼容不同版本的 pymilvus 返回格式
        count = stats.get("row_count", 0)
        st.sidebar.metric("已存储片段数", count)
    except Exception:
        st.sidebar.metric("已存储片段数", "N/A")


def render_chat_tab():
    """
    渲染知识库问答 Tab
    """
    st.header("💬 知识库问答")
    
    st.markdown("""
    在这个页面，你可以向知识库提问。系统会：
    1. 🔍 **检索** - 在向量数据库中找到最相关的文本片段
    2. 🤖 **生成** - 将检索结果作为上下文，让大模型生成回答
    """)
    
    st.divider()
    
    # 初始化聊天历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 如果是助手消息，显示检索到的上下文
            if message["role"] == "assistant" and "contexts" in message:
                with st.expander("🔍 查看检索到的背景知识"):
                    for i, ctx in enumerate(message["contexts"]):
                        st.markdown(f"**片段 {i+1}** (来源: `{ctx['source']}`)")
                        st.text(ctx['text'][:500] + "..." if len(ctx['text']) > 500 else ctx['text'])
                        st.caption(f"相似度: {ctx['distance']:.4f}")
                        st.divider()
    
    # 用户输入
    if prompt := st.chat_input("请输入你的问题..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成助手回复
        with st.chat_message("assistant"):
            # 创建占位符用于显示思考过程
            thinking_placeholder = st.empty()
            
            with thinking_placeholder.container():
                st.info("🤔 正在思考...")
                
                # 步骤 1: 检索相似文本
                st.write("🔍 **步骤 1: 检索相关知识**")
                st.caption(
                    "原理：将你的问题转为向量，"
                    "在数据库中寻找语义最相似的文本片段"
                )
                
                try:
                    contexts = search_similar(prompt, top_k=3)
                    st.write(f"   ✅ 找到 {len(contexts)} 个相关片段")
                    
                    # 显示检索到的片段预览
                    for i, ctx in enumerate(contexts):
                        st.caption(f"   片段 {i+1}: {ctx['source']} (相似度: {ctx['distance']:.4f})")
                    
                except Exception as e:
                    st.error(f"检索失败: {str(e)}")
                    contexts = []
                
                # 步骤 2: 生成回答
                st.write("🤖 **步骤 2: 生成回答**")
                st.caption("调用大模型，结合检索到的上下文生成答案")
            
            # 如果有检索结果，生成回答
            if contexts:
                try:
                    answer = generate_answer(prompt, contexts)
                    
                    # 清除思考过程，显示最终答案
                    thinking_placeholder.empty()
                    st.markdown(answer)
                    
                    # 显示检索上下文（可展开）
                    with st.expander("🔍 查看检索到的背景知识"):
                        st.caption("这些是从知识库中检索到的、用于生成回答的参考文本：")
                        
                        for i, ctx in enumerate(contexts):
                            st.markdown(f"**片段 {i+1}** (来源: `{ctx['source']}` | 相似度: `{ctx['distance']:.4f}`)")
                            # 显示文本片段，限制长度
                            display_text = ctx['text'][:800] + "..." if len(ctx['text']) > 800 else ctx['text']
                            st.text_area(f"内容_{i}", display_text, height=100, label_visibility="collapsed", disabled=True)
                            st.divider()
                    
                    # 保存到历史
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "contexts": contexts
                    })
                    
                except Exception as e:
                    thinking_placeholder.empty()
                    st.error(f"生成回答失败: {str(e)}")
            else:
                thinking_placeholder.empty()
                st.warning("⚠️ 未能在知识库中找到相关信息。请先上传一些文档！")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "⚠️ 未能在知识库中找到相关信息。请先上传一些文档！"
                })

def render_database_tab():
    """
    渲染数据库透视 Tab
    """
    st.header("🔍 数据库透视")
    
    st.markdown("""
    这个页面展示 Milvus 向量数据库中存储的所有文本片段。
    你可以查看已导入的文档内容和元数据信息。
    """)
    
    st.divider()
    
    # 获取数据
    try:
        df = get_all_documents()
        
        if len(df) == 0:
            st.info("📭 数据库为空。请先在左侧上传文档！")
        else:
            # 显示统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总片段数", len(df))
            with col2:
                unique_sources = df['来源文件'].nunique()
                st.metric("来源文件数", unique_sources)
            with col3:
                total_chars = df['字符数'].sum()
                st.metric("总字符数", f"{total_chars:,}")
            
            st.divider()
            
            # 搜索过滤
            search_term = st.text_input("🔍 搜索文本内容", placeholder="输入关键词过滤...")
            
            if search_term:
                filtered_df = df[df['文本片段'].str.contains(search_term, case=False, na=False)]
                st.caption(f"找到 {len(filtered_df)} 条匹配记录")
            else:
                filtered_df = df
            
            # 显示数据表格
            st.subheader("📋 存储的文本片段")
            
            # 限制文本显示长度
            display_df = filtered_df.copy()
            display_df['文本片段'] = display_df['文本片段'].apply(
                lambda x: x[:200] + "..." if len(x) > 200 else x
            )
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "来源文件": st.column_config.TextColumn("来源文件", width="medium"),
                    "文本片段": st.column_config.TextColumn("文本片段", width="large"),
                    "字符数": st.column_config.NumberColumn("字符数", width="small"),
                    "上传时间": st.column_config.TextColumn("上传时间", width="medium"),
                }
            )
            
            # 显示详细信息（可选）
            st.divider()
            st.subheader("📖 查看完整内容")
            
            selected_id = st.selectbox(
                "选择要查看的片段 ID",
                options=filtered_df['ID'].tolist(),
                format_func=lambda x: f"ID: {x} | {filtered_df[filtered_df['ID']==x]['来源文件'].values[0]}"
            )
            
            if selected_id:
                selected_row = filtered_df[filtered_df['ID'] == selected_id].iloc[0]
                st.markdown(f"**来源文件:** `{selected_row['来源文件']}`")
                st.markdown(f"**字符数:** {selected_row['字符数']}")
                st.markdown(f"**上传时间:** {selected_row['上传时间']}")
                st.markdown("**完整内容:**")
                st.text_area("content", selected_row['文本片段'], height=300, label_visibility="collapsed")
                
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")

# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """
    主函数 - 应用入口
    """
    # 页面标题
    st.title("📚 RAG 知识库问答系统")
    st.caption("基于 Ollama + Milvus Lite 的本地知识库检索增强生成演示")
    
    # 初始化系统
    init_status = initialize_system()
    
    # 如果有初始化错误，显示警告
    if init_status["errors"]:
        for error in init_status["errors"]:
            st.error(error)
    
    # 显示系统状态
    col1, col2, col3 = st.columns(3)
    with col1:
        if init_status["milvus_ready"]:
            st.success("✅ Milvus 数据库")
        else:
            st.error("❌ Milvus 数据库")
    with col2:
        if init_status["ollama_ready"]:
            st.success("✅ Ollama 服务")
        else:
            st.error("❌ Ollama 服务")
    with col3:
        if init_status["collection_ready"]:
            st.success(f"✅ Collection: {COLLECTION_NAME}")
        else:
            st.warning(f"⏳ Collection: {COLLECTION_NAME}")
    
    st.divider()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 创建两个 Tab
    tab1, tab2 = st.tabs([
        "💬 知识库问答", 
        "🔍 数据库透视"
    ])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_database_tab()

# 运行主程序
if __name__ == "__main__":
    main()
