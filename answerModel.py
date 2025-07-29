import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime

class AdvancedAnswerModel:
    """高级答案生成模型，专注于生成准确的最终答案"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                 model: str = "qwen2.5-7b-instruct"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # 答案生成历史
        self.answer_history = []
        
        # 初始化答案生成prompt
        self._init_answer_prompts()
    
    def _init_answer_prompts(self):
        """初始化答案生成的prompt模板"""
        
        # 基础答案生成prompt
        self.base_answer_prompt = """
        You are an expert answer generator. Based on the provided information, generate a precise and accurate answer.
        
        Question: {question}
        Context: {context}
        Retrieved Information: {retrieved_info}
        Memory Chunks: {memory_chunks}
        Task Type: {task_type}
        
        Requirements:
        1. Answer must be directly relevant to the question
        2. Use only information provided in the context and retrieved information
        3. Be concise but complete
        4. If information is insufficient, state what is missing
        5. Provide confidence level (High/Medium/Low)
        
        Generate your answer in this format:
        ANSWER: [Your precise answer]
        CONFIDENCE: [High/Medium/Low]
        REASONING: [Brief explanation of how you arrived at this answer]
        """
        
        # 任务特定答案prompt
        self.task_specific_prompts = {
            "multi_hop": """
            For multi-hop reasoning, your answer should:
            1. Show the logical chain of reasoning
            2. Connect information from multiple sources
            3. Clearly state the final conclusion
            
            Question: {question}
            Reasoning Chain: {reasoning_chain}
            Connected Facts: {connected_facts}
            
            MULTI_HOP_ANSWER: [Answer showing the reasoning path]
            REASONING_STEPS: [Step 1] -> [Step 2] -> [Final Answer]
            CONFIDENCE: [High/Medium/Low]
            """,
            
            "summarization": """
            For summarization, your answer should:
            1. Capture the main points
            2. Be appropriately concise
            3. Maintain key information
            
            Content to Summarize: {content}
            Summary Focus: {question}
            Key Points: {key_points}
            
            SUMMARY: [Concise summary addressing the question]
            COVERAGE: [What percentage of key information is included]
            CONFIDENCE: [High/Medium/Low]
            """,
            
            "single_doc_qa": """
            For single document QA, your answer should:
            1. Be directly extracted from or inferred from the document
            2. Be precise and factual
            3. Include relevant details
            
            Document: {document}
            Question: {question}
            Relevant Passages: {relevant_passages}
            
            ANSWER: [Precise answer from the document]
            SOURCE: [Which part of document supports this answer]
            CONFIDENCE: [High/Medium/Low]
            """,
            
            "classification": """
            For classification, your answer should:
            1. Clearly state the category
            2. Explain the classification criteria used
            3. Show confidence in the classification
            
            Text to Classify: {text}
            Classification Task: {question}
            Features Identified: {features}
            
            CATEGORY: [The assigned category]
            FEATURES: [Key features that led to this classification]
            CONFIDENCE: [High/Medium/Low]
            """,
            
            "retrieval": """
            For retrieval tasks, your answer should:
            1. Identify the most relevant passage
            2. Explain why it's most relevant
            3. Provide ranking if multiple passages
            
            Query: {query}
            Passages: {passages}
            Ranking Criteria: {criteria}
            
            BEST_PASSAGE: [Most relevant passage]
            RELEVANCE_SCORE: [0.0-1.0]
            EXPLANATION: [Why this passage is most relevant]
            """,
            
            "code": """
            For code-related tasks, your answer should:
            1. Be technically accurate
            2. Reference specific code elements
            3. Explain code behavior or functionality
            
            Code: {code}
            Question: {question}
            Code Analysis: {analysis}
            
            CODE_ANSWER: [Technical answer about the code]
            CODE_REFERENCE: [Specific lines or functions referenced]
            CONFIDENCE: [High/Medium/Low]
            """,
            
            "counting": """
            For counting tasks, your answer should:
            1. Provide the exact count
            2. Show the counting method
            3. Verify the result
            
            Content: {content}
            Count Target: {question}
            Counting Process: {process}
            
            COUNT: [Exact number]
            METHOD: [How the counting was performed]
            VERIFICATION: [Double-check result]
            """
        }
        
        # 答案验证prompt
        self.answer_verification_prompt = """
        Verify the quality and accuracy of this answer.
        
        Question: {question}
        Generated Answer: {answer}
        Original Context: {context}
        Task Type: {task_type}
        
        Evaluate:
        1. Accuracy: Is the answer factually correct?
        2. Completeness: Does it fully address the question?
        3. Relevance: Is it directly relevant to the question?
        4. Clarity: Is it clear and well-structured?
        
        ACCURACY_SCORE: [0.0-1.0]
        COMPLETENESS_SCORE: [0.0-1.0]
        RELEVANCE_SCORE: [0.0-1.0]
        CLARITY_SCORE: [0.0-1.0]
        OVERALL_SCORE: [0.0-1.0]
        
        ISSUES: [Any problems identified]
        SUGGESTIONS: [How to improve the answer]
        """
        
        # 答案精炼prompt
        self.answer_refinement_prompt = """
        Refine this answer to make it more accurate and concise.
        
        Original Answer: {original_answer}
        Identified Issues: {issues}
        Improvement Suggestions: {suggestions}
        Question: {question}
        
        Generate an improved answer that addresses the issues:
        
        REFINED_ANSWER: [Improved version of the answer]
        IMPROVEMENTS_MADE: [What was changed and why]
        FINAL_CONFIDENCE: [High/Medium/Low]
        """
    
    def call_api(self, prompt: str, max_tokens: int = 600) -> str:
        """调用API生成答案"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert answer generator focused on accuracy and precision."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # 低温度确保一致性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def generate_answer(self, question: str, context: str, retrieved_info: Dict[str, Any],
                       memory_chunks: List[Dict], task_type: str = "general") -> Dict[str, Any]:
        """生成高质量答案"""
        
        # 步骤1: 根据任务类型选择合适的prompt
        if task_type in self.task_specific_prompts:
            answer_result = self._generate_task_specific_answer(
                question, context, retrieved_info, memory_chunks, task_type
            )
        else:
            answer_result = self._generate_base_answer(
                question, context, retrieved_info, memory_chunks, task_type
            )
        
        # 步骤2: 验证答案质量
        verification_result = self._verify_answer_quality(
            question, answer_result["answer"], context, task_type
        )
        
        # 步骤3: 如果质量不够，进行精炼
        if verification_result["overall_score"] < 0.8:
            refined_result = self._refine_answer(
                answer_result["answer"], verification_result, question
            )
            answer_result.update(refined_result)
        
        # 步骤4: 记录答案生成历史
        answer_record = {
            "question": question,
            "answer": answer_result["answer"],
            "task_type": task_type,
            "confidence": answer_result.get("confidence", "Medium"),
            "verification": verification_result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.answer_history.append(answer_record)
        
        return answer_result
    
    def _generate_base_answer(self, question: str, context: str, retrieved_info: Dict[str, Any],
                            memory_chunks: List[Dict], task_type: str) -> Dict[str, Any]:
        """生成基础答案"""
        
        # 格式化输入信息
        memory_text = self._format_memory_chunks(memory_chunks)
        retrieved_text = self._format_retrieved_info(retrieved_info)
        
        # 构建prompt
        prompt = self.base_answer_prompt.format(
            question=question,
            context=context[:800],  # 限制长度
            retrieved_info=retrieved_text,
            memory_chunks=memory_text,
            task_type=task_type
        )
        
        # 生成答案
        response = self.call_api(prompt)
        
        # 解析结果
        return self._parse_answer_response(response)
    
    def _generate_task_specific_answer(self, question: str, context: str, 
                                     retrieved_info: Dict[str, Any], memory_chunks: List[Dict],
                                     task_type: str) -> Dict[str, Any]:
        """生成任务特定答案"""
        
        # 准备任务特定的输入
        task_inputs = self._prepare_task_inputs(
            question, context, retrieved_info, memory_chunks, task_type
        )
        
        # 构建任务特定prompt
        prompt = self.task_specific_prompts[task_type].format(**task_inputs)
        
        # 生成答案
        response = self.call_api(prompt)
        
        # 解析任务特定结果
        return self._parse_task_specific_response(response, task_type)
    
    def _prepare_task_inputs(self, question: str, context: str, retrieved_info: Dict[str, Any],
                           memory_chunks: List[Dict], task_type: str) -> Dict[str, str]:
        """准备任务特定的输入"""
        
        base_inputs = {
            "question": question,
            "context": context[:600],
            "content": context[:600]
        }
        
        if task_type == "multi_hop":
            base_inputs.update({
                "reasoning_chain": retrieved_info.get("task_specific_results", {}).get("reasoning_chain", ""),
                "connected_facts": self._extract_connected_facts(memory_chunks)
            })
        
        elif task_type == "summarization":
            base_inputs.update({
                "key_points": self._extract_key_points(retrieved_info, memory_chunks)
            })
        
        elif task_type == "single_doc_qa":
            base_inputs.update({
                "document": context[:800],
                "relevant_passages": self._extract_relevant_passages(retrieved_info)
            })
        
        elif task_type == "classification":
            base_inputs.update({
                "text": context[:600],
                "features": self._extract_classification_features(retrieved_info)
            })
        
        elif task_type == "retrieval":
            base_inputs.update({
                "query": question,
                "passages": context[:800],
                "criteria": self._extract_ranking_criteria(retrieved_info)
            })
        
        elif task_type == "code":
            base_inputs.update({
                "code": context[:800],
                "analysis": retrieved_info.get("task_specific_results", {}).get("code_analysis", "")
            })
        
        elif task_type == "counting":
            base_inputs.update({
                "process": retrieved_info.get("task_specific_results", {}).get("counting_process", "")
            })
        
        return base_inputs
    
    def _verify_answer_quality(self, question: str, answer: str, context: str, 
                             task_type: str) -> Dict[str, Any]:
        """验证答案质量"""
        
        prompt = self.answer_verification_prompt.format(
            question=question,
            answer=answer,
            context=context[:500],
            task_type=task_type
        )
        
        response = self.call_api(prompt)
        return self._parse_verification_response(response)
    
    def _refine_answer(self, original_answer: str, verification_result: Dict[str, Any],
                      question: str) -> Dict[str, Any]:
        """精炼答案"""
        
        prompt = self.answer_refinement_prompt.format(
            original_answer=original_answer,
            issues=verification_result.get("issues", "No specific issues"),
            suggestions=verification_result.get("suggestions", "No specific suggestions"),
            question=question
        )
        
        response = self.call_api(prompt)
        return self._parse_refinement_response(response)
    
    # 辅助方法
    def _format_memory_chunks(self, memory_chunks: List[Dict]) -> str:
        """格式化记忆块"""
        if not memory_chunks:
            return "No memory chunks available."
        
        formatted = []
        for chunk in memory_chunks[:5]:  # 最多5个
            formatted.append(f"- {chunk.get('content', 'N/A')}")
        
        return "\n".join(formatted)
    
    def _format_retrieved_info(self, retrieved_info: Dict[str, Any]) -> str:
        """格式化检索信息"""
        if not retrieved_info:
            return "No retrieved information available."
        
        formatted = []
        
        # 融合上下文
        fused_context = retrieved_info.get("fused_context", "")
        if fused_context:
            formatted.append(f"Fused Context: {fused_context[:300]}")
        
        # 任务特定结果
        task_results = retrieved_info.get("task_specific_results", {})
        if task_results:
            for key, value in task_results.items():
                if isinstance(value, str) and value:
                    formatted.append(f"{key}: {value[:200]}")
        
        return "\n".join(formatted) if formatted else "No specific retrieved information."
    
    def _extract_connected_facts(self, memory_chunks: List[Dict]) -> str:
        """提取连接的事实"""
        facts = [chunk["content"] for chunk in memory_chunks if chunk.get("type") == "facts"]
        return " | ".join(facts[:3]) if facts else "No connected facts."
    
    def _extract_key_points(self, retrieved_info: Dict[str, Any], memory_chunks: List[Dict]) -> str:
        """提取关键点"""
        # 从检索信息和记忆中提取关键点
        key_points = []
        
        # 从记忆块中提取
        for chunk in memory_chunks:
            if "key" in chunk.get("content", "").lower():
                key_points.append(chunk["content"])
        
        return " | ".join(key_points[:3]) if key_points else "No key points identified."
    
    def _extract_relevant_passages(self, retrieved_info: Dict[str, Any]) -> str:
        """提取相关段落"""
        task_results = retrieved_info.get("task_specific_results", {})
        relevant_info = task_results.get("relevant_info", "")
        return relevant_info[:300] if relevant_info else "No relevant passages identified."
    
    def _extract_classification_features(self, retrieved_info: Dict[str, Any]) -> str:
        """提取分类特征"""
        task_results = retrieved_info.get("task_specific_results", {})
        features = task_results.get("classification_features", "")
        return features[:200] if features else "No classification features identified."
    
    def _extract_ranking_criteria(self, retrieved_info: Dict[str, Any]) -> str:
        """提取排序标准"""
        task_results = retrieved_info.get("task_specific_results", {})
        ranking = task_results.get("passage_ranking", "")
        return ranking[:200] if ranking else "No ranking criteria identified."
    
    # 解析方法
    def _parse_answer_response(self, response: str) -> Dict[str, Any]:
        """解析基础答案响应"""
        result = {"answer": "", "confidence": "Medium", "reasoning": ""}
        
        # 提取答案
        answer_match = re.search(r"ANSWER:\s*(.+?)(?=\n|$)", response, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        
        # 提取置信度
        confidence_match = re.search(r"CONFIDENCE:\s*(.+?)(?=\n|$)", response)
        if confidence_match:
            result["confidence"] = confidence_match.group(1).strip()
        
        # 提取推理
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?=\n|$)", response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        
        # 如果没有找到结构化答案，使用整个响应
        if not result["answer"]:
            result["answer"] = response.strip()
        
        return result
    
    def _parse_task_specific_response(self, response: str, task_type: str) -> Dict[str, Any]:
        """解析任务特定响应"""
        result = {"answer": "", "confidence": "Medium"}
        
        # 根据任务类型提取不同字段
        if task_type == "multi_hop":
            answer_match = re.search(r"MULTI_HOP_ANSWER:\s*(.+?)(?=\n|$)", response, re.DOTALL)
            steps_match = re.search(r"REASONING_STEPS:\s*(.+?)(?=\n|$)", response, re.DOTALL)
            
            if answer_match:
                result["answer"] = answer_match.group(1).strip()
            if steps_match:
                result["reasoning_steps"] = steps_match.group(1).strip()
        
        elif task_type == "summarization":
            summary_match = re.search(r"SUMMARY:\s*(.+?)(?=\n|$)", response, re.DOTALL)
            coverage_match = re.search(r"COVERAGE:\s*(.+?)(?=\n|$)", response)
            
            if summary_match:
                result["answer"] = summary_match.group(1).strip()
            if coverage_match:
                result["coverage"] = coverage_match.group(1).strip()
        
        elif task_type == "classification":
            category_match = re.search(r"CATEGORY:\s*(.+?)(?=\n|$)", response)
            features_match = re.search(r"FEATURES:\s*(.+?)(?=\n|$)", response, re.DOTALL)
            
            if category_match:
                result["answer"] = category_match.group(1).strip()
            if features_match:
                result["features"] = features_match.group(1).strip()
        
        elif task_type == "counting":
            count_match = re.search(r"COUNT:\s*(.+?)(?=\n|$)", response)
            method_match = re.search(r"METHOD:\s*(.+?)(?=\n|$)", response, re.DOTALL)
            
            if count_match:
                result["answer"] = count_match.group(1).strip()
            if method_match:
                result["method"] = method_match.group(1).strip()
        
        # 通用置信度提取
        confidence_match = re.search(r"CONFIDENCE:\s*(.+?)(?=\n|$)", response)
        if confidence_match:
            result["confidence"] = confidence_match.group(1).strip()
        
        # 如果没有找到特定答案，使用整个响应
        if not result["answer"]:
            result["answer"] = response.strip()
        
        return result
    
    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """解析验证响应"""
        result = {
            "accuracy_score": 0.8,
            "completeness_score": 0.8,
            "relevance_score": 0.8,
            "clarity_score": 0.8,
            "overall_score": 0.8,
            "issues": "",
            "suggestions": ""
        }
        
        # 提取各种分数
        score_patterns = {
            "accuracy_score": r"ACCURACY_SCORE:\s*([0-9.]+)",
            "completeness_score": r"COMPLETENESS_SCORE:\s*([0-9.]+)",
            "relevance_score": r"RELEVANCE_SCORE:\s*([0-9.]+)",
            "clarity_score": r"CLARITY_SCORE:\s*([0-9.]+)",
            "overall_score": r"OVERALL_SCORE:\s*([0-9.]+)"
        }
        
        for key, pattern in score_patterns.items():
            match = re.search(pattern, response)
            if match:
                try:
                    result[key] = float(match.group(1))
                except ValueError:
                    pass
        
        # 提取问题和建议
        issues_match = re.search(r"ISSUES:\s*(.+?)(?=\n[A-Z]|$)", response, re.DOTALL)
        if issues_match:
            result["issues"] = issues_match.group(1).strip()
        
        suggestions_match = re.search(r"SUGGESTIONS:\s*(.+?)(?=\n[A-Z]|$)", response, re.DOTALL)
        if suggestions_match:
            result["suggestions"] = suggestions_match.group(1).strip()
        
        return result
    
    def _parse_refinement_response(self, response: str) -> Dict[str, Any]:
        """解析精炼响应"""
        result = {"answer": "", "confidence": "Medium", "improvements": ""}
        
        # 提取精炼后的答案
        answer_match = re.search(r"REFINED_ANSWER:\s*(.+?)(?=\n[A-Z]|$)", response, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        
        # 提取改进说明
        improvements_match = re.search(r"IMPROVEMENTS_MADE:\s*(.+?)(?=\n[A-Z]|$)", response, re.DOTALL)
        if improvements_match:
            result["improvements"] = improvements_match.group(1).strip()
        
        # 提取最终置信度
        confidence_match = re.search(r"FINAL_CONFIDENCE:\s*(.+?)(?=\n|$)", response)
        if confidence_match:
            result["confidence"] = confidence_match.group(1).strip()
        
        return result
    
    def get_answer_statistics(self) -> Dict[str, Any]:
        """获取答案生成统计"""
        if not self.answer_history:
            return {"total_answers": 0}
        
        total = len(self.answer_history)
        high_confidence = sum(1 for a in self.answer_history if a["confidence"] == "High")
        
        task_types = {}
        for answer in self.answer_history:
            task_type = answer["task_type"]
            task_types[task_type] = task_types.get(task_type, 0) + 1
        
        avg_overall_score = sum(
            a["verification"]["overall_score"] for a in self.answer_history
        ) / total
        
        return {
            "total_answers": total,
            "high_confidence_rate": high_confidence / total,
            "task_type_distribution": task_types,
            "average_quality_score": avg_overall_score,
            "last_generated": self.answer_history[-1]["timestamp"] if self.answer_history else None
        }

def main():
    """测试高级答案模型"""
    answer_model = AdvancedAnswerModel(
        api_key="your-api-key-here"
    )
    
    print("=== 测试高级答案模型 ===")
    
    # 测试基础答案生成
    question = "Who wrote Harry Potter?"
    context = "Harry Potter is a series of fantasy novels written by British author J. K. Rowling."
    retrieved_info = {"fused_context": "J.K. Rowling is the author of Harry Potter series."}
    memory_chunks = [{"content": "J.K. Rowling wrote Harry Potter", "type": "facts"}]
    
    print("1. 生成基础答案...")
    result = answer_model.generate_answer(
        question, context, retrieved_info, memory_chunks, "single_doc_qa"
    )
    
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']}")
    
    # 测试多跳推理答案
    print("\n2. 生成多跳推理答案...")
    multi_hop_question = "What is the nationality of the Harry Potter author?"
    multi_hop_retrieved = {
        "task_specific_results": {
            "reasoning_chain": "Step 1: Identify author -> Step 2: Find nationality"
        }
    }
    
    multi_hop_result = answer_model.generate_answer(
        multi_hop_question, context, multi_hop_retrieved, memory_chunks, "multi_hop"
    )
    
    print(f"多跳答案: {multi_hop_result['answer']}")
    print(f"置信度: {multi_hop_result['confidence']}")
    
    # 显示统计
    stats = answer_model.get_answer_statistics()
    print(f"\n答案生成统计: {stats}")

if __name__ == "__main__":
    main()