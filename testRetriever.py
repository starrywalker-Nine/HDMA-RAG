import re
from typing import List, Tuple, Dict, Any
from openai import OpenAI
from datasets import load_dataset
import json
from time import sleep
from datetime import datetime

class OptimizedMultiTurnRetriever:
    """优化的多轮对话检索系统，集成改进的记忆管理和答案生成"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                 model: str = "qwen2.5-7b-instruct"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # 导入优化的组件
        from testQA import DialogueQASystem
        from testChunks import OptimizedMemoryChunkExtractor
        from answerModel import AdvancedAnswerModel
        
        self.qa_system = DialogueQASystem(api_key, base_url, model)
        self.memory_extractor = OptimizedMemoryChunkExtractor(api_key, base_url, model)
        self.answer_model = AdvancedAnswerModel(api_key, base_url, model)
        
        # 会话管理
        self.current_session = {
            "session_id": self._generate_session_id(),
            "start_time": datetime.now().isoformat(),
            "turns": [],
            "task_type": None,
            "performance_metrics": {
                "total_processing_time": 0,
                "memory_efficiency": 0,
                "answer_quality": 0
            }
        }
        
        # 初始化精简的检索prompt
        self._init_streamlined_prompts()
    
    def _init_streamlined_prompts(self):
        """初始化精简的检索prompt"""
        
        # 精简的检索策略prompt
        self.streamlined_retrieval_prompt = """
        Determine the best retrieval approach for this question.
        
        Question: {question}
        Task Type: {task_type}
        Available Memory: {memory_summary}
        Context: {context}
        
        Provide a focused retrieval strategy:
        APPROACH: [direct/multi-step/memory-based]
        KEY_INFO_NEEDED: [what information is required]
        MEMORY_RELEVANCE: [how relevant is existing memory]
        CONFIDENCE: [high/medium/low]
        """
        
        # 上下文融合prompt
        self.context_fusion_prompt = """
        Combine information efficiently for answering the question.
        
        Question: {question}
        Context: {context}
        Memory: {memory}
        Retrieved: {retrieved}
        
        Create a focused information summary:
        KEY_FACTS: [essential facts for answering]
        SUPPORTING_INFO: [additional supporting information]
        GAPS: [any missing information]
        """
    
    def call_api(self, prompt: str, max_tokens: int = 600) -> str:
        """优化的API调用"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an efficient information retrieval assistant. Be concise and focused."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def process_longbench_sample(self, sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """优化的LongBench样本处理流程"""
        
        start_time = datetime.now()
        
        # 确定任务类型
        task_type = self._determine_task_type(dataset_name)
        self.current_session["task_type"] = task_type
        
        # 提取样本信息
        question = sample.get('input', '')
        context = sample.get('context', '')
        ground_truth = sample.get('answers', [''])[0] if sample.get('answers') else ''
        
        print(f"处理 {dataset_name} 样本: {question[:100]}...")
        
        try:
            # 步骤1: 初始QA对话（精简）
            print("步骤1: 初始QA对话")
            initial_qa = self.qa_system.task_adapted_qa(question, context, task_type)
            
            # 步骤2: 精简的记忆块抽取
            print("步骤2: 抽取记忆块")
            memory_chunks = self.memory_extractor.extract_memory_chunks(
                question, initial_qa.get('answer', ''), context, task_type
            )
            
            # 步骤3: 智能检索策略
            print("步骤3: 执行检索策略")
            retrieval_results = self._execute_streamlined_retrieval(
                question, context, memory_chunks, task_type
            )
            
            # 步骤4: 高质量答案生成
            print("步骤4: 生成最终答案")
            final_answer_result = self.answer_model.generate_answer(
                question, context, retrieval_results, 
                self._format_memory_for_answer(memory_chunks), task_type
            )
            
            # 计算处理时间
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 构建结果
            result = {
                "turn_id": len(self.current_session["turns"]) + 1,
                "question": question,
                "context": context,
                "ground_truth": ground_truth,
                "task_type": task_type,
                "initial_qa": initial_qa,
                "memory_chunks": memory_chunks,
                "retrieval_results": retrieval_results,
                "final_answer": final_answer_result["answer"],
                "answer_confidence": final_answer_result.get("confidence", "Medium"),
                "answer_reasoning": final_answer_result.get("reasoning", ""),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": self._calculate_performance_metrics(
                    memory_chunks, retrieval_results, final_answer_result, processing_time
                )
            }
            
            self.current_session["turns"].append(result)
            return result
            
        except Exception as e:
            print(f"处理样本时出错: {e}")
            return {
                "turn_id": len(self.current_session["turns"]) + 1,
                "question": question,
                "context": context,
                "ground_truth": ground_truth,
                "task_type": task_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_streamlined_retrieval(self, question: str, context: str, 
                                     memory_chunks: Dict[str, List[Dict]], 
                                     task_type: str) -> Dict[str, Any]:
        """执行精简的检索策略"""
        
        # 生成记忆摘要
        memory_summary = self._generate_memory_summary(memory_chunks)
        
        # 确定检索策略
        strategy_prompt = self.streamlined_retrieval_prompt.format(
            question=question,
            task_type=task_type,
            memory_summary=memory_summary,
            context=context[:400]
        )
        
        strategy_response = self.call_api(strategy_prompt, max_tokens=300)
        strategy = self._parse_retrieval_strategy(strategy_response)
        
        # 查找相关记忆
        relevant_memory = self.memory_extractor.find_relevant_chunks(question, context, top_k=3)
        
        # 执行任务特定检索
        task_specific_results = self._execute_task_specific_retrieval(
            question, context, relevant_memory, task_type
        )
        
        # 融合信息
        fused_info = self._fuse_information(
            question, context, memory_summary, task_specific_results
        )
        
        return {
            "strategy": strategy,
            "relevant_memory": relevant_memory,
            "task_specific_results": task_specific_results,
            "fused_information": fused_info
        }
    
    def _execute_task_specific_retrieval(self, question: str, context: str,
                                       relevant_memory: List[Dict], task_type: str) -> Dict[str, Any]:
        """执行任务特定的检索"""
        
        if task_type == "multi_hop":
            return self._multi_hop_retrieval(question, context, relevant_memory)
        elif task_type == "summarization":
            return self._summarization_retrieval(question, context, relevant_memory)
        elif task_type == "single_doc_qa":
            return self._single_doc_retrieval(question, context, relevant_memory)
        elif task_type == "classification":
            return self._classification_retrieval(question, context, relevant_memory)
        elif task_type == "retrieval":
            return self._passage_retrieval(question, context, relevant_memory)
        elif task_type == "code":
            return self._code_retrieval(question, context, relevant_memory)
        elif task_type == "counting":
            return self._counting_retrieval(question, context, relevant_memory)
        else:
            return self._general_retrieval(question, context, relevant_memory)
    
    def _multi_hop_retrieval(self, question: str, context: str, memory: List[Dict]) -> Dict[str, Any]:
        """多跳推理检索"""
        prompt = f"""
        Multi-hop reasoning for: {question}
        Context: {context[:400]}
        Memory: {self._format_memory_list(memory)}
        
        Identify the reasoning chain:
        STEP1: [first reasoning step]
        STEP2: [second reasoning step]
        CONCLUSION: [final conclusion]
        """
        
        result = self.call_api(prompt, max_tokens=400)
        return {"reasoning_chain": result}
    
    def _summarization_retrieval(self, question: str, context: str, memory: List[Dict]) -> Dict[str, Any]:
        """摘要检索"""
        prompt = f"""
        Summarization task: {question}
        Content: {context[:500]}
        
        Identify key elements:
        MAIN_POINTS: [3-5 main points]
        STRUCTURE: [how to organize]
        """
        
        result = self.call_api(prompt, max_tokens=300)
        return {"summary_elements": result}
    
    def _single_doc_retrieval(self, question: str, context: str, memory: List[Dict]) -> Dict[str, Any]:
        """单文档检索"""
        prompt = f"""
        Single document QA: {question}
        Document: {context[:500]}
        
        Find relevant information:
        RELEVANT_PARTS: [specific parts that answer the question]
        KEY_FACTS: [key facts from the document]
        """
        
        result = self.call_api(prompt, max_tokens=300)
        return {"relevant_info": result}
    
    def _classification_retrieval(self, question: str, context: str, memory: List[Dict]) -> Dict[str, Any]:
        """分类检索"""
        prompt = f"""
        Classification task: {question}
        Text: {context[:400]}
        
        Identify classification features:
        KEY_FEATURES: [discriminative features]
        CATEGORY_INDICATORS: [indicators for classification]
        """
        
        result = self.call_api(prompt, max_tokens=250)
        return {"classification_features": result}
    
    def _passage_retrieval(self, question: str, context: str, memory: List[Dict]) -> Dict[str, Any]:
        """段落检索"""
        prompt = f"""
        Passage retrieval: {question}
        Passages: {context[:500]}
        
        Rank passages:
        BEST_MATCH: [most relevant passage]
        RELEVANCE_REASON: [why it's most relevant]
        """
        
        result = self.call_api(prompt, max_tokens=250)
        return {"passage_ranking": result}
    
    def _code_retrieval(self, question: str, context: str, memory: List[Dict]) -> Dict[str, Any]:
        """代码检索"""
        prompt = f"""
        Code analysis: {question}
        Code: {context[:500]}
        
        Analyze code:
        FUNCTIONALITY: [what the code does]
        RELEVANT_PARTS: [parts relevant to question]
        """
        
        result = self.call_api(prompt, max_tokens=300)
        return {"code_analysis": result}
    
    def _counting_retrieval(self, question: str, context: str, memory: List[Dict]) -> Dict[str, Any]:
        """计数检索"""
        prompt = f"""
        Counting task: {question}
        Content: {context[:400]}
        
        Count systematically:
        ITEMS_TO_COUNT: [what to count]
        COUNTING_METHOD: [how to count]
        RESULT: [count result]
        """
        
        result = self.call_api(prompt, max_tokens=200)
        return {"counting_process": result}
    
    def _general_retrieval(self, question: str, context: str, memory: List[Dict]) -> Dict[str, Any]:
        """通用检索"""
        prompt = f"""
        General QA: {question}
        Context: {context[:400]}
        
        Extract relevant information for answering the question.
        """
        
        result = self.call_api(prompt, max_tokens=300)
        return {"general_info": result}
    
    def _fuse_information(self, question: str, context: str, memory_summary: str,
                         task_results: Dict[str, Any]) -> str:
        """融合信息"""
        
        prompt = self.context_fusion_prompt.format(
            question=question,
            context=context[:300],
            memory=memory_summary,
            retrieved=str(task_results)[:300]
        )
        
        return self.call_api(prompt, max_tokens=400)
    
    def _format_memory_for_answer(self, memory_chunks: Dict[str, List[Dict]]) -> List[Dict]:
        """为答案生成格式化记忆块"""
        formatted_memory = []
        
        for chunk_type, chunks in memory_chunks.items():
            for chunk in chunks:
                formatted_memory.append({
                    "content": chunk.get("content", ""),
                    "type": chunk_type,
                    "relevance_score": chunk.get("relevance_score", 1.0)
                })
        
        return formatted_memory
    
    def _generate_memory_summary(self, memory_chunks: Dict[str, List[Dict]]) -> str:
        """生成记忆摘要"""
        if not memory_chunks or not any(memory_chunks.values()):
            return "No memory available."
        
        summary_parts = []
        for chunk_type, chunks in memory_chunks.items():
            if chunks:
                summary_parts.append(f"{chunk_type.upper()}: {len(chunks)} items")
                for chunk in chunks[:2]:  # 最多显示2个
                    content = chunk.get("content", "")[:100]
                    summary_parts.append(f"  - {content}")
        
        return "\n".join(summary_parts)
    
    def _format_memory_list(self, memory_list: List[Dict]) -> str:
        """格式化记忆列表"""
        if not memory_list:
            return "No memory available."
        
        formatted = []
        for memory in memory_list[:3]:  # 最多3个
            content = memory.get("content", "")[:100]
            formatted.append(f"- {content}")
        
        return "\n".join(formatted)
    
    def _parse_retrieval_strategy(self, strategy_text: str) -> Dict[str, str]:
        """解析检索策略"""
        strategy = {
            "approach": "direct",
            "key_info_needed": "",
            "memory_relevance": "medium",
            "confidence": "medium"
        }
        
        # 简单的解析
        lines = strategy_text.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key in strategy:
                    strategy[key] = value
        
        return strategy
    
    def _calculate_performance_metrics(self, memory_chunks: Dict[str, List[Dict]], 
                                     retrieval_results: Dict[str, Any],
                                     answer_result: Dict[str, Any], 
                                     processing_time: float) -> Dict[str, Any]:
        """计算性能指标"""
        
        # 记忆效率
        total_memory = sum(len(chunks) for chunks in memory_chunks.values())
        memory_efficiency = min(1.0, 5.0 / max(total_memory, 1))  # 理想记忆块数量为5
        
        # 答案质量（基于置信度）
        confidence_map = {"High": 1.0, "Medium": 0.7, "Low": 0.4}
        answer_quality = confidence_map.get(answer_result.get("confidence", "Medium"), 0.7)
        
        # 处理效率
        processing_efficiency = max(0.1, min(1.0, 30.0 / max(processing_time, 1)))  # 理想处理时间30秒
        
        return {
            "memory_efficiency": memory_efficiency,
            "answer_quality": answer_quality,
            "processing_efficiency": processing_efficiency,
            "total_memory_chunks": total_memory,
            "processing_time": processing_time
        }
    
    def _determine_task_type(self, dataset_name: str) -> str:
        """确定任务类型"""
        task_mapping = {
            "hotpotqa": "multi_hop", "2wikimqa": "multi_hop", "musique": "multi_hop",
            "narrativeqa": "single_doc_qa", "qasper": "single_doc_qa", 
            "multifieldqa_en": "single_doc_qa", "multifieldqa_zh": "single_doc_qa", "dureader": "single_doc_qa",
            "gov_report": "summarization", "qmsum": "summarization", "multi_news": "summarization", "vcsum": "summarization",
            "trec": "classification", "lsht": "classification", "samsum": "summarization",
            "passage_retrieval_en": "retrieval", "passage_retrieval_zh": "retrieval",
            "lcc": "code", "repobench-p": "code",
            "passage_count": "counting", "triviaqa": "single_doc_qa"
        }
        return task_mapping.get(dataset_name, "single_doc_qa")
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        if not self.current_session["turns"]:
            return {"session_id": self.current_session["session_id"], "total_turns": 0}
        
        # 计算平均性能指标
        turns = self.current_session["turns"]
        avg_processing_time = sum(t.get("processing_time", 0) for t in turns) / len(turns)
        
        performance_metrics = [t.get("performance_metrics", {}) for t in turns if "performance_metrics" in t]
        if performance_metrics:
            avg_memory_efficiency = sum(p.get("memory_efficiency", 0) for p in performance_metrics) / len(performance_metrics)
            avg_answer_quality = sum(p.get("answer_quality", 0) for p in performance_metrics) / len(performance_metrics)
        else:
            avg_memory_efficiency = 0
            avg_answer_quality = 0
        
        return {
            "session_id": self.current_session["session_id"],
            "total_turns": len(turns),
            "task_types": list(set(t.get("task_type") for t in turns)),
            "average_processing_time": avg_processing_time,
            "average_memory_efficiency": avg_memory_efficiency,
            "average_answer_quality": avg_answer_quality,
            "start_time": self.current_session["start_time"],
            "end_time": datetime.now().isoformat()
        }
    
    def reset_session(self):
        """重置会话"""
        self.current_session = {
            "session_id": self._generate_session_id(),
            "start_time": datetime.now().isoformat(),
            "turns": [],
            "task_type": None,
            "performance_metrics": {
                "total_processing_time": 0,
                "memory_efficiency": 0,
                "answer_quality": 0
            }
        }
        self.qa_system.reset_dialogue()
        self.memory_extractor.clear_memory()

def main():
    """测试优化的多轮检索系统"""
    retriever = OptimizedMultiTurnRetriever(
        api_key="your-api-key-here"
    )
    
    print("=== 测试优化的多轮检索系统 ===")
    
    # 测试样本
    sample = {
        "input": "Who wrote Harry Potter and what is the author's nationality?",
        "context": "Harry Potter is a series of fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends.",
        "answers": ["J.K. Rowling, British"]
    }
    
    # 处理样本
    result = retriever.process_longbench_sample(sample, "multifieldqa_en")
    
    print(f"\n=== 处理结果 ===")
    print(f"最终答案: {result.get('final_answer', 'N/A')}")
    print(f"答案置信度: {result.get('answer_confidence', 'N/A')}")
    print(f"处理时间: {result.get('processing_time', 0):.2f}秒")
    
    # 显示性能指标
    metrics = result.get('performance_metrics', {})
    print(f"\n=== 性能指标 ===")
    print(f"记忆效率: {metrics.get('memory_efficiency', 0):.3f}")
    print(f"答案质量: {metrics.get('answer_quality', 0):.3f}")
    print(f"处理效率: {metrics.get('processing_efficiency', 0):.3f}")
    
    # 会话摘要
    summary = retriever.get_session_summary()
    print(f"\n=== 会话摘要 ===")
    print(f"会话ID: {summary['session_id']}")
    print(f"总轮数: {summary['total_turns']}")
    print(f"平均处理时间: {summary['average_processing_time']:.2f}秒")

if __name__ == "__main__":
    main()