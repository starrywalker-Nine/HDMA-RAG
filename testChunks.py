import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from openai import OpenAI

class OptimizedMemoryChunkExtractor:
    """优化的记忆块抽取器，改进记忆存储和管理"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                 model: str = "qwen2.5-7b-instruct"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # 优化的记忆存储结构
        self.memory_store = {
            "facts": {},           # {chunk_id: fact_data}
            "actions": {},         # {chunk_id: action_data}
            "reasoning": {},       # {chunk_id: reasoning_data}
            "relations": {},       # {chunk_id: relation_data}
            "index": {             # 索引结构
                "by_content": {},  # {content_hash: chunk_id}
                "by_topic": {},    # {topic: [chunk_ids]}
                "by_relevance": {} # {query_hash: [ranked_chunk_ids]}
            }
        }
        
        self.chunk_counter = 0
        self._init_prompts()
    
    def _init_prompts(self):
        """初始化精简的记忆块抽取prompt"""
        
        # 精简的记忆块抽取prompt
        self.simplified_extraction_prompt = """
        Extract key memory chunks from the dialogue. Focus on essential information only.
        
        Dialogue:
        Question: {question}
        Answer: {answer}
        Context: {context}
        Task Type: {task_type}
        
        Extract ONLY the most important information in this format:
        
        FACTS:
        F1: [One key fact from the dialogue]
        F2: [Another key fact if relevant]
        
        ACTIONS:
        A1: [User's main intent/query]
        A2: [Any follow-up actions needed]
        
        REASONING:
        R1: [Key reasoning step used]
        R2: [Important logical connection]
        
        Keep it concise - maximum 2-3 items per category.
        """
        
        # 记忆检索prompt
        self.memory_retrieval_prompt = """
        Find the most relevant memory chunks for this question.
        
        Question: {question}
        Context: {context}
        
        Available Memory:
        {memory_summary}
        
        Return the top 3 most relevant chunks with their relevance scores (0.0-1.0):
        
        RELEVANT:
        chunk_id: score: reason
        """
        
        # 记忆更新prompt
        self.memory_update_prompt = """
        Update memory based on new information. Be selective - only update if truly necessary.
        
        New Information:
        Question: {question}
        Answer: {answer}
        
        Existing Relevant Memory:
        {existing_memory}
        
        Should any memory be updated? If yes, specify:
        UPDATE: chunk_id: new_content
        
        Should new memory be created? If yes, specify:
        NEW: type: content
        
        If no updates needed, respond: NO_UPDATES
        """
    
    def call_api(self, prompt: str, max_tokens: int = 800) -> str:
        """调用API，减少token使用"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise memory extraction assistant. Keep responses brief and focused."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def extract_memory_chunks(self, question: str, answer: str, context: str, 
                            task_type: str = "general") -> Dict[str, List[Dict]]:
        """精简的记忆块抽取"""
        
        # 构建精简prompt
        prompt = self.simplified_extraction_prompt.format(
            question=question,
            answer=answer,
            context=context[:500],  # 限制上下文长度
            task_type=task_type
        )
        
        # 调用API抽取
        extraction_result = self.call_api(prompt)
        
        # 解析并存储
        chunks = self._parse_simplified_extraction(extraction_result)
        self._store_chunks_efficiently(chunks, question, task_type)
        
        return chunks
    
    def _parse_simplified_extraction(self, extraction_text: str) -> Dict[str, List[Dict]]:
        """解析精简的抽取结果"""
        chunks = {
            "facts": [],
            "actions": [],
            "reasoning": [],
            "relations": []
        }
        
        current_section = None
        lines = extraction_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 识别section
            if line.startswith("FACTS:"):
                current_section = "facts"
                continue
            elif line.startswith("ACTIONS:"):
                current_section = "actions"
                continue
            elif line.startswith("REASONING:"):
                current_section = "reasoning"
                continue
            
            # 解析内容
            if current_section and line and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    chunk_id = f"{current_section[0].upper()}{self.chunk_counter}"
                    self.chunk_counter += 1
                    
                    chunk = {
                        "chunk_id": chunk_id,
                        "content": parts[1].strip(),
                        "type": current_section,
                        "timestamp": datetime.now().isoformat(),
                        "relevance_score": 1.0
                    }
                    
                    chunks[current_section].append(chunk)
        
        return chunks
    
    def _store_chunks_efficiently(self, chunks: Dict[str, List[Dict]], 
                                 question: str, task_type: str):
        """高效存储记忆块"""
        
        for chunk_type, chunk_list in chunks.items():
            for chunk in chunk_list:
                chunk_id = chunk["chunk_id"]
                
                # 存储到对应类型
                if chunk_type == "facts":
                    self.memory_store["facts"][chunk_id] = chunk
                elif chunk_type == "actions":
                    self.memory_store["actions"][chunk_id] = chunk
                elif chunk_type == "reasoning":
                    self.memory_store["reasoning"][chunk_id] = chunk
                
                # 更新索引
                content_hash = hash(chunk["content"])
                self.memory_store["index"]["by_content"][content_hash] = chunk_id
                
                # 按主题索引
                topic = self._extract_topic(chunk["content"], task_type)
                if topic not in self.memory_store["index"]["by_topic"]:
                    self.memory_store["index"]["by_topic"][topic] = []
                self.memory_store["index"]["by_topic"][topic].append(chunk_id)
    
    def find_relevant_chunks(self, question: str, context: str = "", top_k: int = 3) -> List[Dict]:
        """快速查找相关记忆块"""
        
        # 如果没有记忆，直接返回
        if not any(self.memory_store[key] for key in ["facts", "actions", "reasoning"]):
            return []
        
        # 生成记忆摘要
        memory_summary = self._generate_memory_summary()
        
        # 构建检索prompt
        prompt = self.memory_retrieval_prompt.format(
            question=question,
            context=context[:300],
            memory_summary=memory_summary
        )
        
        # 调用API查找
        retrieval_result = self.call_api(prompt)
        
        # 解析结果
        relevant_chunks = self._parse_retrieval_result(retrieval_result, top_k)
        
        return relevant_chunks
    
    def _generate_memory_summary(self) -> str:
        """生成记忆摘要"""
        summary_parts = []
        
        # 事实摘要
        if self.memory_store["facts"]:
            facts = list(self.memory_store["facts"].values())[:3]  # 最多3个
            summary_parts.append("FACTS:")
            for fact in facts:
                summary_parts.append(f"  {fact['chunk_id']}: {fact['content'][:100]}")
        
        # 动作摘要
        if self.memory_store["actions"]:
            actions = list(self.memory_store["actions"].values())[:2]  # 最多2个
            summary_parts.append("ACTIONS:")
            for action in actions:
                summary_parts.append(f"  {action['chunk_id']}: {action['content'][:100]}")
        
        # 推理摘要
        if self.memory_store["reasoning"]:
            reasoning = list(self.memory_store["reasoning"].values())[:2]  # 最多2个
            summary_parts.append("REASONING:")
            for reason in reasoning:
                summary_parts.append(f"  {reason['chunk_id']}: {reason['content'][:100]}")
        
        return "\n".join(summary_parts) if summary_parts else "No memory available."
    
    def _parse_retrieval_result(self, retrieval_text: str, top_k: int) -> List[Dict]:
        """解析检索结果"""
        relevant_chunks = []
        
        lines = retrieval_text.split('\n')
        in_relevant_section = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("RELEVANT:"):
                in_relevant_section = True
                continue
            
            if in_relevant_section and ':' in line:
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    chunk_id = parts[0].strip()
                    try:
                        score = float(parts[1].strip())
                        reason = parts[2].strip()
                        
                        # 查找对应的记忆块
                        chunk = self._find_chunk_by_id(chunk_id)
                        if chunk:
                            chunk_copy = chunk.copy()
                            chunk_copy['relevance_score'] = score
                            chunk_copy['relevance_reason'] = reason
                            relevant_chunks.append(chunk_copy)
                    except ValueError:
                        continue
        
        # 按相关性排序
        relevant_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return relevant_chunks[:top_k]
    
    def _find_chunk_by_id(self, chunk_id: str) -> Dict:
        """根据ID查找记忆块"""
        # 在所有存储中查找
        for store_type in ["facts", "actions", "reasoning", "relations"]:
            if chunk_id in self.memory_store[store_type]:
                return self.memory_store[store_type][chunk_id]
        return None
    
    def update_memory_selectively(self, question: str, answer: str, 
                                relevant_chunks: List[Dict]) -> Dict[str, Any]:
        """选择性更新记忆"""
        
        if not relevant_chunks:
            # 如果没有相关记忆，直接创建新的
            return self.extract_memory_chunks(question, answer, "", "update")
        
        # 格式化现有记忆
        existing_memory = self._format_existing_memory(relevant_chunks)
        
        # 构建更新prompt
        prompt = self.memory_update_prompt.format(
            question=question,
            answer=answer,
            existing_memory=existing_memory
        )
        
        # 调用API判断是否需要更新
        update_result = self.call_api(prompt)
        
        # 解析并应用更新
        updates = self._parse_and_apply_updates(update_result)
        
        return updates
    
    def _format_existing_memory(self, chunks: List[Dict]) -> str:
        """格式化现有记忆"""
        if not chunks:
            return "No existing memory."
        
        formatted = []
        for chunk in chunks:
            formatted.append(f"{chunk['chunk_id']}: {chunk['content']}")
        
        return "\n".join(formatted)
    
    def _parse_and_apply_updates(self, update_text: str) -> Dict[str, Any]:
        """解析并应用更新"""
        updates = {"updated": [], "created": [], "no_changes": False}
        
        if "NO_UPDATES" in update_text:
            updates["no_changes"] = True
            return updates
        
        lines = update_text.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith("UPDATE:"):
                # 更新现有记忆
                parts = line.replace("UPDATE:", "").split(':', 1)
                if len(parts) == 2:
                    chunk_id = parts[0].strip()
                    new_content = parts[1].strip()
                    
                    chunk = self._find_chunk_by_id(chunk_id)
                    if chunk:
                        chunk['content'] = new_content
                        chunk['last_updated'] = datetime.now().isoformat()
                        updates["updated"].append(chunk_id)
            
            elif line.startswith("NEW:"):
                # 创建新记忆
                parts = line.replace("NEW:", "").split(':', 1)
                if len(parts) == 2:
                    chunk_type = parts[0].strip()
                    content = parts[1].strip()
                    
                    new_chunk = {
                        "chunk_id": f"{chunk_type[0].upper()}{self.chunk_counter}",
                        "content": content,
                        "type": chunk_type,
                        "timestamp": datetime.now().isoformat(),
                        "relevance_score": 1.0
                    }
                    
                    self.chunk_counter += 1
                    
                    # 存储新记忆
                    if chunk_type in self.memory_store:
                        self.memory_store[chunk_type][new_chunk["chunk_id"]] = new_chunk
                        updates["created"].append(new_chunk["chunk_id"])
        
        return updates
    
    def _extract_topic(self, content: str, task_type: str) -> str:
        """提取主题用于索引"""
        # 简化的主题提取
        if task_type == "multi_hop":
            return "reasoning"
        elif task_type == "summarization":
            return "summary"
        elif task_type == "classification":
            return "category"
        else:
            return "general"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            "total_chunks": sum(len(store) for store in [
                self.memory_store["facts"],
                self.memory_store["actions"], 
                self.memory_store["reasoning"],
                self.memory_store["relations"]
            ]),
            "facts_count": len(self.memory_store["facts"]),
            "actions_count": len(self.memory_store["actions"]),
            "reasoning_count": len(self.memory_store["reasoning"]),
            "relations_count": len(self.memory_store["relations"]),
            "topics_indexed": len(self.memory_store["index"]["by_topic"]),
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_memory(self):
        """清空记忆"""
        self.memory_store = {
            "facts": {},
            "actions": {},
            "reasoning": {},
            "relations": {},
            "index": {
                "by_content": {},
                "by_topic": {},
                "by_relevance": {}
            }
        }
        self.chunk_counter = 0

def main():
    """测试优化的记忆块抽取器"""
    extractor = OptimizedMemoryChunkExtractor(
        api_key="your-api-key-here"
    )
    
    print("=== 测试优化的记忆块抽取器 ===")
    
    # 测试记忆抽取
    question = "Who wrote Harry Potter?"
    answer = "J.K. Rowling wrote the Harry Potter series."
    context = "Harry Potter is a fantasy novel series."
    
    print("1. 抽取记忆块...")
    chunks = extractor.extract_memory_chunks(question, answer, context, "single_doc_qa")
    
    stats = extractor.get_memory_stats()
    print(f"记忆统计: {stats}")
    
    # 测试记忆检索
    print("\n2. 查找相关记忆...")
    relevant = extractor.find_relevant_chunks("What is J.K. Rowling's nationality?")
    print(f"找到 {len(relevant)} 个相关记忆块")
    
    # 测试记忆更新
    print("\n3. 更新记忆...")
    updates = extractor.update_memory_selectively(
        "What is her nationality?",
        "J.K. Rowling is British.",
        relevant
    )
    print(f"更新结果: {updates}")
    
    final_stats = extractor.get_memory_stats()
    print(f"最终统计: {final_stats}")

if __name__ == "__main__":
    main()