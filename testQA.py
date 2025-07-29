import json
import re
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from openai import OpenAI
from time import sleep

class DialogueQASystem:
    """对话式问答系统，支持多轮对话和历史记录管理"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                 model: str = "qwen2.5-7b-instruct"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dialogue_history = []
        self.memory_chunks = []
        
        # 初始化prompt模板
        self._init_prompts()
    
    def _init_prompts(self):
        """初始化各种prompt模板"""
        
        # 初始QA对话prompt
        self.initial_qa_prompt = """
        You are an intelligent assistant that can answer questions based on given context. 
        Please provide a comprehensive answer to the question using the provided context.
        
        Context: {context}
        Question: {question}
        
        Please provide your answer and explain your reasoning process.
        Answer:
        """
        
        # 历史对话生成prompt
        self.history_generation_prompt = """
        Based on the previous question-answer interaction, generate a structured dialogue history that captures:
        1. The original question intent
        2. Key information extracted from the context
        3. The reasoning process used
        4. The final answer provided
        
        Previous Q&A:
        Question: {question}
        Answer: {answer}
        Context: {context}
        
        Generate a structured dialogue history in the following format:
        
        Dialogue History:
        - User Intent: [What the user was trying to find out]
        - Key Information: [Important facts extracted from context]
        - Reasoning Process: [How the answer was derived]
        - Answer: [The final answer]
        - Confidence: [High/Medium/Low based on context quality]
        """
        
        # 多轮对话prompt
        self.multi_turn_prompt = """
        You are continuing a conversation. Based on the dialogue history and new information, 
        please answer the follow-up question.
        
        Dialogue History:
        {dialogue_history}
        
        New Context (if any): {new_context}
        New Question: {new_question}
        
        Please provide an answer that considers the previous conversation context.
        Answer:
        """
        
        # 任务特定适配prompt
        self.task_adaptation_prompts = {
            "multi_hop": """
            This is a multi-hop reasoning task. You need to:
            1. Identify multiple pieces of information needed
            2. Connect information across different sources
            3. Provide step-by-step reasoning
            
            Question: {question}
            Context: {context}
            
            Please provide step-by-step reasoning and final answer.
            """,
            
            "summarization": """
            This is a summarization task. Please:
            1. Identify key points from the context
            2. Organize information logically
            3. Provide a concise summary
            
            Content to summarize: {context}
            Specific focus: {question}
            
            Summary:
            """,
            
            "single_doc_qa": """
            This is a single document QA task. Please:
            1. Carefully read the document
            2. Extract relevant information
            3. Provide accurate answer
            
            Document: {context}
            Question: {question}
            
            Answer:
            """,
            
            "classification": """
            This is a classification task. Please:
            1. Analyze the given text
            2. Consider the classification criteria
            3. Provide the most appropriate category
            
            Text: {context}
            Classification task: {question}
            
            Category:
            """,
            
            "retrieval": """
            This is a retrieval task. Please:
            1. Understand what information is being sought
            2. Search through the provided passages
            3. Identify the most relevant passage
            
            Query: {question}
            Passages: {context}
            
            Most relevant passage:
            """,
            
            "code": """
            This is a code-related task. Please:
            1. Understand the code context
            2. Analyze the specific question
            3. Provide accurate code-related answer
            
            Code context: {context}
            Question: {question}
            
            Answer:
            """,
            
            "counting": """
            This is a counting task. Please:
            1. Carefully count the requested items
            2. Double-check your count
            3. Provide the exact number
            
            Content: {context}
            Count what: {question}
            
            Count:
            """
        }
    
    def call_api(self, prompt: str, max_tokens: int = 1000) -> str:
        """调用API进行文本生成"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def initial_qa_dialogue(self, question: str, context: str) -> Dict[str, Any]:
        """进行初始QA对话"""
        prompt = self.initial_qa_prompt.format(question=question, context=context)
        answer = self.call_api(prompt)
        
        qa_result = {
            "question": question,
            "context": context,
            "answer": answer,
            "timestamp": self._get_timestamp(),
            "turn_id": len(self.dialogue_history) + 1
        }
        
        return qa_result
    
    def generate_dialogue_history(self, qa_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成结构化的对话历史"""
        prompt = self.history_generation_prompt.format(
            question=qa_result["question"],
            answer=qa_result["answer"],
            context=qa_result["context"]
        )
        
        history_text = self.call_api(prompt)
        
        # 解析历史信息
        history_data = self._parse_dialogue_history(history_text)
        history_data.update({
            "turn_id": qa_result["turn_id"],
            "timestamp": qa_result["timestamp"],
            "raw_history": history_text
        })
        
        # 添加到对话历史
        self.dialogue_history.append(history_data)
        
        return history_data
    
    def _parse_dialogue_history(self, history_text: str) -> Dict[str, str]:
        """解析对话历史文本"""
        history_data = {
            "user_intent": "",
            "key_information": "",
            "reasoning_process": "",
            "answer": "",
            "confidence": "Medium"
        }
        
        # 使用正则表达式提取信息
        patterns = {
            "user_intent": r"User Intent:\s*(.+?)(?=\n-|\n\n|$)",
            "key_information": r"Key Information:\s*(.+?)(?=\n-|\n\n|$)",
            "reasoning_process": r"Reasoning Process:\s*(.+?)(?=\n-|\n\n|$)",
            "answer": r"Answer:\s*(.+?)(?=\n-|\n\n|$)",
            "confidence": r"Confidence:\s*(.+?)(?=\n-|\n\n|$)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, history_text, re.DOTALL | re.IGNORECASE)
            if match:
                history_data[key] = match.group(1).strip()
        
        return history_data
    
    def multi_turn_dialogue(self, new_question: str, new_context: str = "") -> Dict[str, Any]:
        """进行多轮对话"""
        # 格式化对话历史
        history_text = self._format_dialogue_history()
        
        prompt = self.multi_turn_prompt.format(
            dialogue_history=history_text,
            new_context=new_context,
            new_question=new_question
        )
        
        answer = self.call_api(prompt)
        
        # 创建新的QA结果
        qa_result = {
            "question": new_question,
            "context": new_context,
            "answer": answer,
            "timestamp": self._get_timestamp(),
            "turn_id": len(self.dialogue_history) + 1,
            "is_multi_turn": True
        }
        
        return qa_result
    
    def task_adapted_qa(self, question: str, context: str, task_type: str) -> Dict[str, Any]:
        """根据任务类型进行适配的问答"""
        if task_type in self.task_adaptation_prompts:
            prompt = self.task_adaptation_prompts[task_type].format(
                question=question, 
                context=context
            )
        else:
            # 默认使用初始QA prompt
            prompt = self.initial_qa_prompt.format(question=question, context=context)
        
        answer = self.call_api(prompt)
        
        qa_result = {
            "question": question,
            "context": context,
            "answer": answer,
            "task_type": task_type,
            "timestamp": self._get_timestamp(),
            "turn_id": len(self.dialogue_history) + 1
        }
        
        return qa_result
    
    def _format_dialogue_history(self) -> str:
        """格式化对话历史为文本"""
        if not self.dialogue_history:
            return "No previous dialogue history."
        
        formatted_history = []
        for i, history in enumerate(self.dialogue_history, 1):
            formatted_history.append(f"Turn {i}:")
            formatted_history.append(f"  User Intent: {history.get('user_intent', 'N/A')}")
            formatted_history.append(f"  Key Information: {history.get('key_information', 'N/A')}")
            formatted_history.append(f"  Answer: {history.get('answer', 'N/A')}")
            formatted_history.append("")
        
        return "\n".join(formatted_history)
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_dialogue_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        return {
            "total_turns": len(self.dialogue_history),
            "dialogue_history": self.dialogue_history,
            "memory_chunks_count": len(self.memory_chunks)
        }
    
    def reset_dialogue(self):
        """重置对话历史"""
        self.dialogue_history = []
        self.memory_chunks = []

def determine_task_type(dataset_name: str) -> str:
    """根据数据集名称确定任务类型"""
    task_mapping = {
        # 多跳推理
        "hotpotqa": "multi_hop",
        "2wikimqa": "multi_hop", 
        "musique": "multi_hop",
        
        # 单文档QA
        "narrativeqa": "single_doc_qa",
        "qasper": "single_doc_qa",
        "multifieldqa_en": "single_doc_qa",
        "multifieldqa_zh": "single_doc_qa",
        "dureader": "single_doc_qa",
        
        # 摘要任务
        "gov_report": "summarization",
        "qmsum": "summarization", 
        "multi_news": "summarization",
        "vcsum": "summarization",
        
        # 分类任务
        "trec": "classification",
        "triviaqa": "single_doc_qa",  # 虽然名字是trivia，但实际是QA
        "samsum": "summarization",   # 对话摘要
        "lsht": "classification",
        
        # 检索任务
        "passage_retrieval_en": "retrieval",
        "passage_retrieval_zh": "retrieval",
        
        # 代码任务
        "lcc": "code",
        "repobench-p": "code",
        
        # 计数任务
        "passage_count": "counting"
    }
    
    return task_mapping.get(dataset_name, "single_doc_qa")

def main():
    """主函数 - 演示对话式QA系统"""
    # 初始化系统
    qa_system = DialogueQASystem(
        api_key="your-api-key-here"
    )
    
    # 示例：多轮对话流程
    print("=== 对话式QA系统演示 ===")
    
    # 第一轮：初始QA
    question1 = "Who wrote Harry Potter?"
    context1 = "Harry Potter is a series of fantasy novels written by British author J. K. Rowling."
    
    print(f"第一轮问题: {question1}")
    qa_result1 = qa_system.initial_qa_dialogue(question1, context1)
    print(f"回答: {qa_result1['answer']}")
    
    # 生成对话历史
    history1 = qa_system.generate_dialogue_history(qa_result1)
    print(f"对话历史生成完成")
    
    # 第二轮：多轮对话
    question2 = "What is her nationality?"
    print(f"\n第二轮问题: {question2}")
    qa_result2 = qa_system.multi_turn_dialogue(question2)
    print(f"回答: {qa_result2['answer']}")
    
    # 生成第二轮历史
    history2 = qa_system.generate_dialogue_history(qa_result2)
    
    # 第三轮：任务适配QA
    question3 = "Summarize what we know about J.K. Rowling"
    print(f"\n第三轮问题 (摘要任务): {question3}")
    qa_result3 = qa_system.task_adapted_qa(question3, context1, "summarization")
    print(f"回答: {qa_result3['answer']}")
    
    # 显示对话摘要
    summary = qa_system.get_dialogue_summary()
    print(f"\n=== 对话摘要 ===")
    print(f"总轮数: {summary['total_turns']}")
    print(f"记忆块数量: {summary['memory_chunks_count']}")

if __name__ == "__main__":
    main()
