"""
HDMARAG (Hierarchical Declarative Memory Augment RAG) 核心算法
整合了分层记忆管理、step-back抽象思考和增强检索功能
优化支持本地vLLM部署和A100显卡
"""

import json
import re
import math
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
from local_model_interface import load_local_model, LocalModel

class HDMARAGCore:
    """HDMARAG核心算法类 - 优化支持本地vLLM部署"""
    
    def __init__(self, model_name: str = "qwen2.5-7b-instruct", use_local: bool = True):
        self.model_name = model_name
        self.use_local = use_local
        
        # 初始化本地模型
        if use_local:
            self.local_model = load_local_model(model_name)
            print(f"已加载本地模型: {model_name}")
        else:
            self.local_model = None
        
        # 分层记忆存储结构
        self.hierarchical_memory = {
            "episodic": {},      # 情景记忆：具体的问答对话
            "semantic": {},      # 语义记忆：抽象的概念和关系
            "procedural": {},    # 程序记忆：推理过程和方法
            "meta": {},          # 元记忆：关于记忆本身的信息
            "index": {           # 多层索引结构
                "concept_hierarchy": {},  # 概念层次
                "temporal_index": {},     # 时间索引
                "relevance_graph": {},    # 相关性图
                "abstraction_levels": {}  # 抽象层级
            }
        }
        
        # Step-back抽象思考组件
        self.abstraction_engine = AbstractionEngine(self.local_model) if use_local else None
        
        # 增强检索组件
        self.enhanced_retriever = EnhancedRetriever(self.local_model) if use_local else None
        
        # 长上下文分块器
        self.context_chunker = HierarchicalChunker()
        
        # 记忆计数器
        self.memory_counter = 0
        
        # 初始化prompt模板
        self._init_hdmarag_prompts()
    
    def _init_hdmarag_prompts(self):
        """初始化HDMARAG专用prompt模板"""
        
        # 分层记忆抽取prompt
        self.hierarchical_memory_prompt = """
        Extract hierarchical memory from the dialogue using HDMARAG framework.
        
        Dialogue Context:
        Question: {question}
        Answer: {answer}
        Context: {context}
        Task Type: {task_type}
        
        Extract memory at different abstraction levels:
        
        EPISODIC (具体事件):
        E1: [Specific factual information from this dialogue]
        E2: [Concrete details and examples]
        
        SEMANTIC (抽象概念):
        S1: [General concepts and relationships]
        S2: [Abstract principles derived]
        
        PROCEDURAL (推理过程):
        P1: [Reasoning methods used]
        P2: [Problem-solving strategies]
        
        META (元认知):
        M1: [Information about the reasoning process itself]
        M2: [Confidence and uncertainty indicators]
        
        ABSTRACTION_LEVEL: [1-5, where 1=concrete, 5=highly abstract]
        """
        
        # Step-back抽象思考prompt
        self.step_back_prompt = """
        Perform step-back abstraction to elevate thinking dimension and depth.
        
        Current Question: {question}
        Current Context: {context}
        
        Step-back Analysis:
        1. What is the fundamental concept behind this question?
        2. What general principles apply here?
        3. What higher-level patterns can we identify?
        4. How does this relate to broader knowledge domains?
        
        FUNDAMENTAL_CONCEPT: [Core concept]
        GENERAL_PRINCIPLES: [Applicable principles]
        HIGHER_PATTERNS: [Abstract patterns]
        KNOWLEDGE_DOMAINS: [Related domains]
        ABSTRACTION_CHAIN: [Concrete -> Abstract progression]
        """
        
        # 增强检索融合prompt
        self.enhanced_fusion_prompt = """
        Fuse information from multiple sources using HDMARAG enhancement.
        
        Original Question: {question}
        Generated Chunks: {generated_chunks}
        Original Embedding Chunks: {embedding_chunks}
        Hierarchical Memory: {hierarchical_memory}
        Step-back Insights: {step_back_insights}
        
        Perform enhanced fusion:
        1. Compare generated chunks with embedding chunks
        2. Identify complementary information
        3. Resolve conflicts using hierarchical memory
        4. Integrate step-back insights for deeper understanding
        
        FUSED_INFORMATION: [Integrated comprehensive information]
        ENHANCEMENT_FACTORS: [How each source contributed]
        CONFIDENCE_ASSESSMENT: [Overall confidence level]
        REASONING_CHAIN: [Step-by-step reasoning process]
        """
    
    def process_long_context(self, question: str, long_context: str, task_type: str = "general") -> Dict[str, Any]:
        """处理长上下文的主要HDMARAG流程"""
        
        start_time = datetime.now()
        
        try:
            # 步骤1: 分层分块 - 从长上下文生成chunks
            print("步骤1: 分层分块处理")
            hierarchical_chunks = self.context_chunker.generate_hierarchical_chunks(
                long_context, question, task_type
            )
            
            # 步骤2: Step-back抽象思考
            print("步骤2: Step-back抽象思考")
            step_back_insights = self.abstraction_engine.perform_step_back_analysis(
                question, long_context, task_type
            )
            
            # 步骤3: 分层记忆检索
            print("步骤3: 分层记忆检索")
            relevant_memory = self._retrieve_hierarchical_memory(
                question, step_back_insights, task_type
            )
            
            # 步骤4: 增强检索和融合
            print("步骤4: 增强检索和融合")
            enhanced_results = self.enhanced_retriever.enhanced_retrieve_and_fuse(
                question, hierarchical_chunks, relevant_memory, step_back_insights
            )
            
            # 步骤5: 生成最终答案
            print("步骤5: 生成增强答案")
            final_answer = self._generate_enhanced_answer(
                question, enhanced_results, task_type
            )
            
            # 步骤6: 更新分层记忆
            print("步骤6: 更新分层记忆")
            memory_updates = self._update_hierarchical_memory(
                question, final_answer, enhanced_results, step_back_insights, task_type
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "method": "HDMARAG",
                "question": question,
                "task_type": task_type,
                "hierarchical_chunks": hierarchical_chunks,
                "step_back_insights": step_back_insights,
                "relevant_memory": relevant_memory,
                "enhanced_results": enhanced_results,
                "final_answer": final_answer["answer"],
                "answer_confidence": final_answer.get("confidence", "Medium"),
                "reasoning_chain": final_answer.get("reasoning_chain", []),
                "memory_updates": memory_updates,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": self._calculate_hdmarag_metrics(
                    hierarchical_chunks, enhanced_results, processing_time
                )
            }
            
        except Exception as e:
            return {
                "method": "HDMARAG",
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _retrieve_hierarchical_memory(self, question: str, step_back_insights: Dict[str, Any], 
                                    task_type: str) -> Dict[str, List[Dict]]:
        """检索分层记忆"""
        
        relevant_memory = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "meta": []
        }
        
        # 基于step-back insights检索不同层级的记忆
        fundamental_concept = step_back_insights.get("fundamental_concept", "")
        general_principles = step_back_insights.get("general_principles", "")
        
        # 检索语义记忆（概念相关）
        for memory_id, memory in self.hierarchical_memory["semantic"].items():
            if self._is_conceptually_relevant(memory["content"], fundamental_concept):
                memory["relevance_score"] = self._calculate_semantic_relevance(
                    memory["content"], fundamental_concept
                )
                relevant_memory["semantic"].append(memory)
        
        # 检索程序记忆（方法相关）
        for memory_id, memory in self.hierarchical_memory["procedural"].items():
            if self._is_procedurally_relevant(memory["content"], general_principles, task_type):
                memory["relevance_score"] = self._calculate_procedural_relevance(
                    memory["content"], general_principles
                )
                relevant_memory["procedural"].append(memory)
        
        # 检索情景记忆（具体事实）
        for memory_id, memory in self.hierarchical_memory["episodic"].items():
            if self._is_episodically_relevant(memory["content"], question):
                memory["relevance_score"] = self._calculate_episodic_relevance(
                    memory["content"], question
                )
                relevant_memory["episodic"].append(memory)
        
        # 排序并限制数量
        for memory_type in relevant_memory:
            relevant_memory[memory_type].sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )
            relevant_memory[memory_type] = relevant_memory[memory_type][:3]  # 每类最多3个
        
        return relevant_memory
    
    def _generate_enhanced_answer(self, question: str, enhanced_results: Dict[str, Any], 
                                task_type: str) -> Dict[str, Any]:
        """生成增强答案"""
        
        prompt = f"""
        Generate an enhanced answer using HDMARAG methodology.
        
        Question: {question}
        Task Type: {task_type}
        
        Enhanced Information:
        {enhanced_results.get('fused_information', '')}
        
        Reasoning Chain:
        {enhanced_results.get('reasoning_chain', '')}
        
        Enhancement Factors:
        {enhanced_results.get('enhancement_factors', '')}
        
        Generate a comprehensive answer that:
        1. Directly addresses the question
        2. Uses the enhanced information effectively
        3. Shows clear reasoning process
        4. Indicates confidence level
        
        ANSWER: [Your comprehensive answer]
        REASONING_CHAIN: [Step-by-step reasoning]
        CONFIDENCE: [High/Medium/Low]
        ENHANCEMENT_USED: [How enhancement improved the answer]
        """
        
        response = self.call_api(prompt)
        return self._parse_enhanced_answer(response)
    
    def _update_hierarchical_memory(self, question: str, answer_result: Dict[str, Any], 
                                  enhanced_results: Dict[str, Any], step_back_insights: Dict[str, Any],
                                  task_type: str) -> Dict[str, Any]:
        """更新分层记忆"""
        
        # 构建记忆抽取prompt
        prompt = self.hierarchical_memory_prompt.format(
            question=question,
            answer=answer_result.get("answer", ""),
            context=enhanced_results.get("fused_information", "")[:500],
            task_type=task_type
        )
        
        # 抽取分层记忆
        memory_extraction = self.call_api(prompt)
        parsed_memory = self._parse_hierarchical_memory(memory_extraction)
        
        # 存储到分层记忆结构
        updates = {"created": [], "updated": []}
        
        for memory_type, memories in parsed_memory.items():
            if memory_type in self.hierarchical_memory:
                for memory in memories:
                    memory_id = f"{memory_type}_{self.memory_counter}"
                    self.memory_counter += 1
                    
                    memory_obj = {
                        "memory_id": memory_id,
                        "content": memory["content"],
                        "type": memory_type,
                        "abstraction_level": memory.get("abstraction_level", 3),
                        "source_question": question,
                        "task_type": task_type,
                        "timestamp": datetime.now().isoformat(),
                        "step_back_context": step_back_insights
                    }
                    
                    self.hierarchical_memory[memory_type][memory_id] = memory_obj
                    updates["created"].append(memory_id)
        
        # 更新索引
        self._update_memory_indices(parsed_memory, step_back_insights)
        
        return updates
    
    def call_api(self, prompt: str, max_tokens: int = 800) -> str:
        """调用本地模型API"""
        try:
            if self.use_local and self.local_model:
                # 构建完整prompt
                system_prompt = "You are an advanced HDMARAG system with hierarchical memory and step-back reasoning capabilities."
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                response = self.local_model.generate(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    stop=["<|im_end|>", "<|endoftext|>"]
                )
                return response.strip()
            else:
                print("本地模型未加载，无法生成响应")
                return ""
        except Exception as e:
            print(f"本地模型调用失败: {e}")
            return ""
    
    # 辅助方法
    def _is_conceptually_relevant(self, memory_content: str, concept: str) -> bool:
        """判断概念相关性"""
        if not concept:
            return False
        concept_words = set(concept.lower().split())
        memory_words = set(memory_content.lower().split())
        return len(concept_words.intersection(memory_words)) > 0
    
    def _calculate_semantic_relevance(self, memory_content: str, concept: str) -> float:
        """计算语义相关性分数"""
        if not concept:
            return 0.0
        concept_words = set(concept.lower().split())
        memory_words = set(memory_content.lower().split())
        intersection = concept_words.intersection(memory_words)
        union = concept_words.union(memory_words)
        return len(intersection) / len(union) if union else 0.0
    
    def _is_procedurally_relevant(self, memory_content: str, principles: str, task_type: str) -> bool:
        """判断程序相关性"""
        if task_type in memory_content.lower():
            return True
        if principles and any(word in memory_content.lower() for word in principles.lower().split()):
            return True
        return False
    
    def _calculate_procedural_relevance(self, memory_content: str, principles: str) -> float:
        """计算程序相关性分数"""
        if not principles:
            return 0.5
        principle_words = set(principles.lower().split())
        memory_words = set(memory_content.lower().split())
        intersection = principle_words.intersection(memory_words)
        return len(intersection) / len(principle_words) if principle_words else 0.0
    
    def _is_episodically_relevant(self, memory_content: str, question: str) -> bool:
        """判断情景相关性"""
        question_words = set(question.lower().split())
        memory_words = set(memory_content.lower().split())
        return len(question_words.intersection(memory_words)) >= 2
    
    def _calculate_episodic_relevance(self, memory_content: str, question: str) -> float:
        """计算情景相关性分数"""
        question_words = set(question.lower().split())
        memory_words = set(memory_content.lower().split())
        intersection = question_words.intersection(memory_words)
        return len(intersection) / len(question_words) if question_words else 0.0
    
    def _parse_hierarchical_memory(self, extraction_text: str) -> Dict[str, List[Dict]]:
        """解析分层记忆抽取结果"""
        memory_types = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "meta": []
        }
        
        current_type = None
        abstraction_level = 3  # 默认抽象级别
        
        lines = extraction_text.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith("EPISODIC"):
                current_type = "episodic"
                continue
            elif line.startswith("SEMANTIC"):
                current_type = "semantic"
                continue
            elif line.startswith("PROCEDURAL"):
                current_type = "procedural"
                continue
            elif line.startswith("META"):
                current_type = "meta"
                continue
            elif line.startswith("ABSTRACTION_LEVEL:"):
                try:
                    abstraction_level = int(line.split(":")[1].strip())
                except:
                    abstraction_level = 3
                continue
            
            if current_type and line and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    memory_types[current_type].append({
                        "content": parts[1].strip(),
                        "abstraction_level": abstraction_level
                    })
        
        return memory_types
    
    def _parse_enhanced_answer(self, response: str) -> Dict[str, Any]:
        """解析增强答案"""
        result = {
            "answer": "",
            "reasoning_chain": [],
            "confidence": "Medium",
            "enhancement_used": ""
        }
        
        # 提取答案
        answer_match = re.search(r"ANSWER:\s*(.+?)(?=\n[A-Z]|$)", response, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        
        # 提取推理链
        reasoning_match = re.search(r"REASONING_CHAIN:\s*(.+?)(?=\n[A-Z]|$)", response, re.DOTALL)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            result["reasoning_chain"] = [step.strip() for step in reasoning_text.split('\n') if step.strip()]
        
        # 提取置信度
        confidence_match = re.search(r"CONFIDENCE:\s*(.+?)(?=\n|$)", response)
        if confidence_match:
            result["confidence"] = confidence_match.group(1).strip()
        
        # 提取增强使用情况
        enhancement_match = re.search(r"ENHANCEMENT_USED:\s*(.+?)(?=\n[A-Z]|$)", response, re.DOTALL)
        if enhancement_match:
            result["enhancement_used"] = enhancement_match.group(1).strip()
        
        if not result["answer"]:
            result["answer"] = response.strip()
        
        return result
    
    def _update_memory_indices(self, parsed_memory: Dict[str, List[Dict]], 
                             step_back_insights: Dict[str, Any]):
        """更新记忆索引"""
        # 更新概念层次索引
        fundamental_concept = step_back_insights.get("fundamental_concept", "")
        if fundamental_concept:
            if fundamental_concept not in self.hierarchical_memory["index"]["concept_hierarchy"]:
                self.hierarchical_memory["index"]["concept_hierarchy"][fundamental_concept] = []
            
            for memory_type, memories in parsed_memory.items():
                for memory in memories:
                    memory_id = f"{memory_type}_{self.memory_counter - 1}"
                    self.hierarchical_memory["index"]["concept_hierarchy"][fundamental_concept].append(memory_id)
        
        # 更新抽象层级索引
        for memory_type, memories in parsed_memory.items():
            for memory in memories:
                level = memory.get("abstraction_level", 3)
                if level not in self.hierarchical_memory["index"]["abstraction_levels"]:
                    self.hierarchical_memory["index"]["abstraction_levels"][level] = []
                
                memory_id = f"{memory_type}_{self.memory_counter - 1}"
                self.hierarchical_memory["index"]["abstraction_levels"][level].append(memory_id)
    
    def _calculate_hdmarag_metrics(self, hierarchical_chunks: Dict[str, Any], 
                                 enhanced_results: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """计算HDMARAG性能指标"""
        
        # 分层效率
        chunk_levels = len(hierarchical_chunks.get("levels", []))
        hierarchical_efficiency = min(1.0, chunk_levels / 5.0)  # 理想层级数为5
        
        # 增强效果
        enhancement_factors = enhanced_results.get("enhancement_factors", "")
        enhancement_score = len(enhancement_factors.split()) / 50.0 if enhancement_factors else 0.5
        enhancement_score = min(1.0, enhancement_score)
        
        # 记忆利用率
        total_memory = sum(len(self.hierarchical_memory[key]) for key in ["episodic", "semantic", "procedural", "meta"])
        memory_utilization = min(1.0, total_memory / 20.0)  # 理想记忆数量为20
        
        # 处理效率
        processing_efficiency = max(0.1, min(1.0, 60.0 / max(processing_time, 1)))  # 理想处理时间60秒
        
        return {
            "hierarchical_efficiency": hierarchical_efficiency,
            "enhancement_score": enhancement_score,
            "memory_utilization": memory_utilization,
            "processing_efficiency": processing_efficiency,
            "total_memory_count": total_memory,
            "chunk_levels": chunk_levels,
            "processing_time": processing_time
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return {
            "total_memories": {
                "episodic": len(self.hierarchical_memory["episodic"]),
                "semantic": len(self.hierarchical_memory["semantic"]),
                "procedural": len(self.hierarchical_memory["procedural"]),
                "meta": len(self.hierarchical_memory["meta"])
            },
            "index_stats": {
                "concepts": len(self.hierarchical_memory["index"]["concept_hierarchy"]),
                "abstraction_levels": len(self.hierarchical_memory["index"]["abstraction_levels"])
            },
            "memory_counter": self.memory_counter
        }
    
    def reset_memory(self):
        """重置记忆"""
        self.hierarchical_memory = {
            "episodic": {},
            "semantic": {},
            "procedural": {},
            "meta": {},
            "index": {
                "concept_hierarchy": {},
                "temporal_index": {},
                "relevance_graph": {},
                "abstraction_levels": {}
            }
        }
        self.memory_counter = 0


class AbstractionEngine:
    """Step-back抽象思考引擎"""
    
    def __init__(self, local_model: LocalModel):
        self.local_model = local_model
    
    def perform_step_back_analysis(self, question: str, context: str, task_type: str) -> Dict[str, Any]:
        """执行step-back抽象分析"""
        
        prompt = f"""
        Perform step-back abstraction to elevate thinking dimension and depth.
        
        Current Question: {question}
        Current Context: {context[:500]}
        Task Type: {task_type}
        
        Step-back Analysis:
        1. What is the fundamental concept behind this question?
        2. What general principles apply here?
        3. What higher-level patterns can we identify?
        4. How does this relate to broader knowledge domains?
        5. What abstraction chain can we build?
        
        FUNDAMENTAL_CONCEPT: [Core concept]
        GENERAL_PRINCIPLES: [Applicable principles]
        HIGHER_PATTERNS: [Abstract patterns]
        KNOWLEDGE_DOMAINS: [Related domains]
        ABSTRACTION_CHAIN: [Concrete -> Abstract progression]
        THINKING_DIMENSION: [How this elevates thinking]
        """
        
        try:
            # 构建完整prompt
            system_prompt = "You are an expert in abstract reasoning and step-back thinking."
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            result = self.local_model.generate(
                full_prompt,
                max_tokens=600,
                temperature=0.2,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            
            return self._parse_step_back_result(result)
            
        except Exception as e:
            return {
                "fundamental_concept": "",
                "general_principles": "",
                "higher_patterns": "",
                "knowledge_domains": "",
                "abstraction_chain": "",
                "thinking_dimension": "",
                "error": str(e)
            }
    
    def _parse_step_back_result(self, result: str) -> Dict[str, Any]:
        """解析step-back分析结果"""
        parsed = {
            "fundamental_concept": "",
            "general_principles": "",
            "higher_patterns": "",
            "knowledge_domains": "",
            "abstraction_chain": "",
            "thinking_dimension": ""
        }
        
        patterns = {
            "fundamental_concept": r"FUNDAMENTAL_CONCEPT:\s*(.+?)(?=\n[A-Z]|$)",
            "general_principles": r"GENERAL_PRINCIPLES:\s*(.+?)(?=\n[A-Z]|$)",
            "higher_patterns": r"HIGHER_PATTERNS:\s*(.+?)(?=\n[A-Z]|$)",
            "knowledge_domains": r"KNOWLEDGE_DOMAINS:\s*(.+?)(?=\n[A-Z]|$)",
            "abstraction_chain": r"ABSTRACTION_CHAIN:\s*(.+?)(?=\n[A-Z]|$)",
            "thinking_dimension": r"THINKING_DIMENSION:\s*(.+?)(?=\n[A-Z]|$)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, result, re.DOTALL)
            if match:
                parsed[key] = match.group(1).strip()
        
        return parsed


class EnhancedRetriever:
    """增强检索器"""
    
    def __init__(self, local_model: LocalModel):
        self.local_model = local_model
    
    def enhanced_retrieve_and_fuse(self, question: str, hierarchical_chunks: Dict[str, Any],
                                 relevant_memory: Dict[str, List[Dict]], 
                                 step_back_insights: Dict[str, Any]) -> Dict[str, Any]:
        """增强检索和融合"""
        
        # 格式化输入
        generated_chunks = self._format_hierarchical_chunks(hierarchical_chunks)
        embedding_chunks = self._simulate_embedding_chunks(question, hierarchical_chunks)
        memory_text = self._format_memory(relevant_memory)
        insights_text = self._format_insights(step_back_insights)
        
        prompt = f"""
        Fuse information from multiple sources using HDMARAG enhancement.
        
        Original Question: {question}
        Generated Chunks: {generated_chunks}
        Original Embedding Chunks: {embedding_chunks}
        Hierarchical Memory: {memory_text}
        Step-back Insights: {insights_text}
        
        Perform enhanced fusion:
        1. Compare generated chunks with embedding chunks
        2. Identify complementary information
        3. Resolve conflicts using hierarchical memory
        4. Integrate step-back insights for deeper understanding
        
        FUSED_INFORMATION: [Integrated comprehensive information]
        ENHANCEMENT_FACTORS: [How each source contributed]
        CONFIDENCE_ASSESSMENT: [Overall confidence level]
        REASONING_CHAIN: [Step-by-step reasoning process]
        CHUNK_COMPARISON: [How generated and embedding chunks differ/complement]
        """
        
        try:
            # 构建完整prompt
            system_prompt = "You are an expert information fusion system."
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            result = self.local_model.generate(
                full_prompt,
                max_tokens=800,
                temperature=0.1,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            
            return self._parse_fusion_result(result)
            
        except Exception as e:
            return {
                "fused_information": "",
                "enhancement_factors": "",
                "confidence_assessment": "Low",
                "reasoning_chain": "",
                "chunk_comparison": "",
                "error": str(e)
            }
    
    def _format_hierarchical_chunks(self, hierarchical_chunks: Dict[str, Any]) -> str:
        """格式化分层chunks"""
        if not hierarchical_chunks:
            return "No hierarchical chunks available."
        
        formatted = []
        levels = hierarchical_chunks.get("levels", [])
        for i, level in enumerate(levels):
            formatted.append(f"Level {i+1}: {level.get('content', '')[:200]}")
        
        return "\n".join(formatted)
    
    def _simulate_embedding_chunks(self, question: str, hierarchical_chunks: Dict[str, Any]) -> str:
        """模拟原始embedding chunks"""
        # 这里模拟传统embedding检索的结果
        return f"Traditional embedding chunks for: {question[:100]}..."
    
    def _format_memory(self, relevant_memory: Dict[str, List[Dict]]) -> str:
        """格式化记忆"""
        formatted = []
        for memory_type, memories in relevant_memory.items():
            if memories:
                formatted.append(f"{memory_type.upper()}:")
                for memory in memories[:2]:  # 最多2个
                    formatted.append(f"  - {memory.get('content', '')[:100]}")
        
        return "\n".join(formatted) if formatted else "No relevant memory."
    
    def _format_insights(self, step_back_insights: Dict[str, Any]) -> str:
        """格式化step-back insights"""
        if not step_back_insights:
            return "No step-back insights available."
        
        formatted = []
        for key, value in step_back_insights.items():
            if value and key != "error":
                formatted.append(f"{key.replace('_', ' ').title()}: {value[:100]}")
        
        return "\n".join(formatted) if formatted else "No insights available."
    
    def _parse_fusion_result(self, result: str) -> Dict[str, Any]:
        """解析融合结果"""
        parsed = {
            "fused_information": "",
            "enhancement_factors": "",
            "confidence_assessment": "Medium",
            "reasoning_chain": "",
            "chunk_comparison": ""
        }
        
        patterns = {
            "fused_information": r"FUSED_INFORMATION:\s*(.+?)(?=\n[A-Z]|$)",
            "enhancement_factors": r"ENHANCEMENT_FACTORS:\s*(.+?)(?=\n[A-Z]|$)",
            "confidence_assessment": r"CONFIDENCE_ASSESSMENT:\s*(.+?)(?=\n[A-Z]|$)",
            "reasoning_chain": r"REASONING_CHAIN:\s*(.+?)(?=\n[A-Z]|$)",
            "chunk_comparison": r"CHUNK_COMPARISON:\s*(.+?)(?=\n[A-Z]|$)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, result, re.DOTALL)
            if match:
                parsed[key] = match.group(1).strip()
        
        return parsed


class HierarchicalChunker:
    """分层分块器"""
    
    def __init__(self):
        self.chunk_levels = [2000, 1000, 500, 250, 100]  # 不同层级的chunk大小
    
    def generate_hierarchical_chunks(self, long_context: str, question: str, task_type: str) -> Dict[str, Any]:
        """生成分层chunks"""
        
        hierarchical_chunks = {
            "levels": [],
            "total_levels": len(self.chunk_levels),
            "original_length": len(long_context),
            "question_context": question
        }
        
        # 为每个层级生成chunks
        for level, chunk_size in enumerate(self.chunk_levels):
            level_chunks = self._create_chunks_at_level(long_context, chunk_size, question, task_type)
            
            hierarchical_chunks["levels"].append({
                "level": level + 1,
                "chunk_size": chunk_size,
                "chunks": level_chunks,
                "chunk_count": len(level_chunks),
                "content": self._merge_level_content(level_chunks)
            })
        
        return hierarchical_chunks
    
    def _create_chunks_at_level(self, context: str, chunk_size: int, question: str, task_type: str) -> List[Dict[str, Any]]:
        """在特定层级创建chunks"""
        
        # 基于句子边界分割
        sentences = re.split(r'[.!?。！？]\s*', context)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= chunk_size:
                current_chunk += sentence + ". "
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "length": current_length,
                        "relevance_score": self._calculate_chunk_relevance(current_chunk, question)
                    })
                
                current_chunk = sentence + ". "
                current_length = sentence_length
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "length": current_length,
                "relevance_score": self._calculate_chunk_relevance(current_chunk, question)
            })
        
        # 按相关性排序
        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return chunks
    
    def _calculate_chunk_relevance(self, chunk: str, question: str) -> float:
        """计算chunk与问题的相关性"""
        question_words = set(question.lower().split())
        chunk_words = set(chunk.lower().split())
        
        intersection = question_words.intersection(chunk_words)
        union = question_words.union(chunk_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _merge_level_content(self, chunks: List[Dict[str, Any]]) -> str:
        """合并层级内容"""
        # 取前3个最相关的chunks
        top_chunks = chunks[:3]
        return "\n\n".join([chunk["content"] for chunk in top_chunks])


def main():
    """测试HDMARAG系统"""
    hdmarag = HDMARAGCore(
        api_key="your-api-key-here"
    )
    
    print("=== HDMARAG系统测试 ===")
    
    # 测试长上下文
    long_context = """
    Artificial Intelligence (AI) has revolutionized many aspects of modern life. Machine learning, a subset of AI,
    enables computers to learn and improve from experience without being explicitly programmed. Deep learning,
    which uses neural networks with multiple layers, has been particularly successful in tasks like image recognition
    and natural language processing. The development of large language models like GPT has shown remarkable capabilities
    in understanding and generating human-like text. These models are trained on vast amounts of text data and can
    perform various tasks such as translation, summarization, and question answering. However, challenges remain in
    areas like explainability, bias, and computational efficiency. Researchers continue to work on making AI systems
    more robust, fair, and interpretable. The future of AI holds promise for even more sophisticated applications
    in healthcare, education, transportation, and many other domains.
    """
    
    question = "What are the main challenges in AI development?"
    
    print(f"问题: {question}")
    print(f"长上下文长度: {len(long_context)} 字符")
    
    # 处理长上下文
    result = hdmarag.process_long_context(question, long_context, "single_doc_qa")
    
    print(f"\n=== HDMARAG处理结果 ===")
    print(f"最终答案: {result.get('final_answer', 'N/A')}")
    print(f"答案置信度: {result.get('answer_confidence', 'N/A')}")
    print(f"处理时间: {result.get('processing_time', 0):.2f}秒")
    
    # 显示性能指标
    metrics = result.get('performance_metrics', {})
    print(f"\n=== 性能指标 ===")
    print(f"分层效率: {metrics.get('hierarchical_efficiency', 0):.3f}")
    print(f"增强分数: {metrics.get('enhancement_score', 0):.3f}")
    print(f"记忆利用率: {metrics.get('memory_utilization', 0):.3f}")
    print(f"处理效率: {metrics.get('processing_efficiency', 0):.3f}")
    
    # 显示记忆摘要
    memory_summary = hdmarag.get_memory_summary()
    print(f"\n=== 记忆摘要 ===")
    print(f"总记忆数: {memory_summary['total_memories']}")
    print(f"索引统计: {memory_summary['index_stats']}")


if __name__ == "__main__":
    main()