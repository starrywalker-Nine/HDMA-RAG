"""
HDMARAG集成系统
整合所有组件，提供完整的HDMARAG功能
优化支持本地vLLM部署和A100显卡
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from hdmarag_core import HDMARAGCore
from local_model_interface import get_model_manager, LocalModelManager

class HDMARAGSystem:
    """HDMARAG完整系统 - 优化支持本地vLLM部署"""
    
    def __init__(self, model_name: str = "qwen2.5-7b-instruct",
                 config_path: str = "hdmarag_config.json",
                 use_local: bool = True):
        
        self.model_name = model_name
        self.use_local = use_local
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化本地模型管理器
        if use_local:
            self.model_manager = get_model_manager()
            print(f"GPU信息: {self.model_manager.gpu_info}")
        
        # 初始化核心组件
        self.hdmarag_core = HDMARAGCore(model_name, use_local)
        
        # 系统状态
        self.session_history = []
        self.performance_metrics = {
            "total_processed": 0,
            "average_processing_time": 0,
            "average_enhancement_score": 0,
            "memory_efficiency": 0
        }
        
        print("HDMARAG系统初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"加载配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "hdmarag_config": {
                "enable_hierarchical_memory": True,
                "enable_step_back_reasoning": True,
                "enable_enhanced_retrieval": True,
                "max_memory_per_type": 50,
                "abstraction_levels": 5,
                "chunk_levels": [2000, 1000, 500, 250, 100]
            },
            "processing_config": {
                "max_context_length": 10000,
                "enable_parallel_processing": True,
                "timeout_seconds": 120
            }
        }
    
    def process_sample(self, sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """处理单个样本"""
        
        start_time = datetime.now()
        
        try:
            # 提取样本信息
            question = sample.get('input', '')
            context = sample.get('context', '')
            ground_truth = sample.get('answers', [''])[0] if sample.get('answers') else ''
            
            # 确定任务类型
            task_type = self._determine_task_type(dataset_name)
            
            print(f"处理样本: {dataset_name} - {question[:50]}...")
            
            # 使用HDMARAG核心处理
            hdmarag_result = self.hdmarag_core.process_long_context(question, context, task_type)
            
            # 如果HDMARAG处理失败，使用备用方法
            if "error" in hdmarag_result:
                print("HDMARAG处理失败，使用备用方法")
                backup_result = self._process_with_backup_method(question, context, task_type)
                hdmarag_result.update(backup_result)
            
            # 计算处理时间
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 构建完整结果
            result = {
                "dataset": dataset_name,
                "question": question,
                "context": context,
                "ground_truth": ground_truth,
                "task_type": task_type,
                "hdmarag_result": hdmarag_result,
                "final_answer": hdmarag_result.get("final_answer", ""),
                "confidence": hdmarag_result.get("answer_confidence", "Medium"),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "method": "HDMARAG",
                "success": "error" not in hdmarag_result
            }
            
            # 更新性能指标
            self._update_performance_metrics(result)
            
            # 添加到会话历史
            self.session_history.append(result)
            
            return result
            
        except Exception as e:
            return {
                "dataset": dataset_name,
                "question": question,
                "context": context,
                "ground_truth": ground_truth,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
                "method": "HDMARAG",
                "success": False
            }
    
    def _process_with_backup_method(self, question: str, context: str, task_type: str) -> Dict[str, Any]:
        """使用备用方法处理"""
        
        try:
            # 使用优化的多轮检索器作为备用
            sample = {
                "input": question,
                "context": context,
                "answers": [""]
            }
            
            backup_result = self.retriever.process_longbench_sample(sample, task_type)
            
            return {
                "final_answer": backup_result.get("final_answer", ""),
                "answer_confidence": backup_result.get("answer_confidence", "Low"),
                "backup_method": "OptimizedMultiTurnRetriever",
                "backup_used": True
            }
            
        except Exception as e:
            return {
                "final_answer": f"处理失败: {str(e)}",
                "answer_confidence": "Low",
                "backup_method": "None",
                "backup_used": True,
                "backup_error": str(e)
            }
    
    def process_dataset(self, dataset_name: str, max_samples: int = 10) -> Dict[str, Any]:
        """处理整个数据集"""
        
        print(f"开始处理数据集: {dataset_name}")
        
        # 加载数据集
        dataset_samples = self._load_dataset_samples(dataset_name, max_samples)
        
        if not dataset_samples:
            return {
                "dataset": dataset_name,
                "error": "无法加载数据集",
                "processed_count": 0
            }
        
        results = []
        successful_count = 0
        
        for i, sample in enumerate(dataset_samples):
            print(f"处理样本 {i+1}/{len(dataset_samples)}")
            
            result = self.process_sample(sample, dataset_name)
            results.append(result)
            
            if result.get("success", False):
                successful_count += 1
        
        # 计算数据集级别的统计
        dataset_stats = self._calculate_dataset_stats(results)
        
        return {
            "dataset": dataset_name,
            "total_samples": len(dataset_samples),
            "processed_samples": len(results),
            "successful_samples": successful_count,
            "success_rate": successful_count / len(results) if results else 0,
            "results": results,
            "statistics": dataset_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    def _load_dataset_samples(self, dataset_name: str, max_samples: int) -> List[Dict[str, Any]]:
        """加载数据集样本"""
        
        data_dir = self.config.get("dataset_config", {}).get("data_directory", "./longbench/data")
        file_path = os.path.join(data_dir, f"{dataset_name}.jsonl")
        
        if not os.path.exists(file_path):
            print(f"数据文件不存在: {file_path}")
            return []
        
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    
                    sample = json.loads(line.strip())
                    samples.append(sample)
            
            print(f"成功加载 {len(samples)} 个样本")
            return samples
            
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return []
    
    def _determine_task_type(self, dataset_name: str) -> str:
        """确定任务类型"""
        task_mapping = {
            "narrativeqa": "single_doc_qa",
            "qasper": "single_doc_qa", 
            "multifieldqa_en": "single_doc_qa",
            "multifieldqa_zh": "single_doc_qa",
            "hotpotqa": "multi_hop",
            "2wikimqa": "multi_hop",
            "musique": "multi_hop",
            "gov_report": "summarization",
            "qmsum": "summarization",
            "multi_news": "summarization",
            "vcsum": "summarization",
            "trec": "classification",
            "lsht": "classification",
            "samsum": "summarization",
            "passage_retrieval_en": "retrieval",
            "passage_retrieval_zh": "retrieval",
            "lcc": "code",
            "repobench-p": "code",
            "passage_count": "counting"
        }
        
        return task_mapping.get(dataset_name, "single_doc_qa")
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """更新性能指标"""
        
        self.performance_metrics["total_processed"] += 1
        
        # 更新平均处理时间
        current_time = result.get("processing_time", 0)
        total = self.performance_metrics["total_processed"]
        prev_avg = self.performance_metrics["average_processing_time"]
        self.performance_metrics["average_processing_time"] = (prev_avg * (total - 1) + current_time) / total
        
        # 更新增强分数
        hdmarag_result = result.get("hdmarag_result", {})
        metrics = hdmarag_result.get("performance_metrics", {})
        enhancement_score = metrics.get("enhancement_score", 0)
        
        prev_enhancement = self.performance_metrics["average_enhancement_score"]
        self.performance_metrics["average_enhancement_score"] = (prev_enhancement * (total - 1) + enhancement_score) / total
        
        # 更新记忆效率
        memory_efficiency = metrics.get("memory_utilization", 0)
        prev_memory = self.performance_metrics["memory_efficiency"]
        self.performance_metrics["memory_efficiency"] = (prev_memory * (total - 1) + memory_efficiency) / total
    
    def _calculate_dataset_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算数据集统计"""
        
        if not results:
            return {}
        
        # 处理时间统计
        processing_times = [r.get("processing_time", 0) for r in results]
        avg_time = sum(processing_times) / len(processing_times)
        
        # 置信度分布
        confidence_dist = {"High": 0, "Medium": 0, "Low": 0}
        for result in results:
            conf = result.get("confidence", "Medium")
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        
        # HDMARAG性能指标
        enhancement_scores = []
        memory_utilizations = []
        
        for result in results:
            hdmarag_result = result.get("hdmarag_result", {})
            metrics = hdmarag_result.get("performance_metrics", {})
            
            if "enhancement_score" in metrics:
                enhancement_scores.append(metrics["enhancement_score"])
            if "memory_utilization" in metrics:
                memory_utilizations.append(metrics["memory_utilization"])
        
        return {
            "average_processing_time": avg_time,
            "confidence_distribution": confidence_dist,
            "average_enhancement_score": sum(enhancement_scores) / len(enhancement_scores) if enhancement_scores else 0,
            "average_memory_utilization": sum(memory_utilizations) / len(memory_utilizations) if memory_utilizations else 0,
            "total_samples": len(results),
            "successful_samples": sum(1 for r in results if r.get("success", False))
        }
    
    def evaluate_multiple_datasets(self, dataset_names: List[str], max_samples_per_dataset: int = 10) -> Dict[str, Any]:
        """评估多个数据集"""
        
        print(f"开始评估 {len(dataset_names)} 个数据集")
        
        all_results = {}
        overall_stats = {
            "total_datasets": len(dataset_names),
            "total_samples": 0,
            "total_successful": 0,
            "overall_success_rate": 0,
            "average_processing_time": 0,
            "start_time": datetime.now().isoformat()
        }
        
        for dataset_name in dataset_names:
            print(f"\n{'='*50}")
            print(f"评估数据集: {dataset_name}")
            print(f"{'='*50}")
            
            dataset_result = self.process_dataset(dataset_name, max_samples_per_dataset)
            all_results[dataset_name] = dataset_result
            
            # 更新总体统计
            overall_stats["total_samples"] += dataset_result.get("processed_samples", 0)
            overall_stats["total_successful"] += dataset_result.get("successful_samples", 0)
        
        # 计算总体成功率
        if overall_stats["total_samples"] > 0:
            overall_stats["overall_success_rate"] = overall_stats["total_successful"] / overall_stats["total_samples"]
        
        overall_stats["average_processing_time"] = self.performance_metrics["average_processing_time"]
        overall_stats["end_time"] = datetime.now().isoformat()
        
        return {
            "evaluation_results": all_results,
            "overall_statistics": overall_stats,
            "system_performance": self.performance_metrics,
            "memory_summary": self.hdmarag_core.get_memory_summary()
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "hdmarag_results"):
        """保存结果"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        results_file = os.path.join(output_dir, f"hdmarag_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存摘要报告
        summary_file = os.path.join(output_dir, f"hdmarag_summary_{timestamp}.json")
        summary = {
            "timestamp": timestamp,
            "overall_statistics": results.get("overall_statistics", {}),
            "system_performance": results.get("system_performance", {}),
            "memory_summary": results.get("memory_summary", {})
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_dir}")
        print(f"完整结果: {results_file}")
        print(f"摘要报告: {summary_file}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "performance_metrics": self.performance_metrics,
            "memory_summary": self.hdmarag_core.get_memory_summary(),
            "session_count": len(self.session_history),
            "last_processed": self.session_history[-1]["timestamp"] if self.session_history else None
        }
    
    def reset_system(self):
        """重置系统"""
        self.hdmarag_core.reset_memory()
        self.qa_system.reset_dialogue()
        self.memory_extractor.clear_memory()
        self.retriever.reset_session()
        
        self.session_history = []
        self.performance_metrics = {
            "total_processed": 0,
            "average_processing_time": 0,
            "average_enhancement_score": 0,
            "memory_efficiency": 0
        }
        
        print("HDMARAG系统已重置")


def main():
    """主函数 - 演示HDMARAG系统"""
    
    # 初始化系统
    hdmarag_system = HDMARAGSystem(
        api_key="your-api-key-here"
    )
    
    print("=== HDMARAG系统演示 ===")
    
    # 测试单个样本
    test_sample = {
        "input": "What are the main challenges in artificial intelligence development?",
        "context": """
        Artificial Intelligence (AI) has revolutionized many aspects of modern life. However, several challenges remain in AI development. 
        First, explainability is a major concern - many AI systems, especially deep learning models, operate as "black boxes" where 
        the decision-making process is not transparent. Second, bias in AI systems can lead to unfair outcomes, particularly when 
        training data reflects historical prejudices. Third, computational efficiency remains a challenge as larger models require 
        enormous computational resources. Fourth, robustness and reliability are crucial for deploying AI in critical applications. 
        Finally, ethical considerations around privacy, autonomy, and the impact on employment need careful attention.
        """,
        "answers": ["explainability, bias, computational efficiency, robustness, ethical considerations"]
    }
    
    print("测试单个样本处理...")
    result = hdmarag_system.process_sample(test_sample, "multifieldqa_en")
    
    print(f"\n=== 处理结果 ===")
    print(f"问题: {result['question']}")
    print(f"最终答案: {result['final_answer']}")
    print(f"置信度: {result['confidence']}")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"成功: {result['success']}")
    
    # 显示系统状态
    status = hdmarag_system.get_system_status()
    print(f"\n=== 系统状态 ===")
    print(f"性能指标: {status['performance_metrics']}")
    print(f"记忆摘要: {status['memory_summary']}")
    
    # 测试多数据集评估（如果数据文件存在）
    test_datasets = ["multifieldqa_en", "hotpotqa"]
    print(f"\n=== 测试多数据集评估 ===")
    
    try:
        evaluation_results = hdmarag_system.evaluate_multiple_datasets(test_datasets, max_samples_per_dataset=2)
        
        print("评估完成!")
        print(f"总体统计: {evaluation_results['overall_statistics']}")
        
        # 保存结果
        hdmarag_system.save_results(evaluation_results)
        
    except Exception as e:
        print(f"多数据集评估失败: {e}")
        print("这可能是因为数据文件不存在，请确保longbench数据文件在正确位置")


if __name__ == "__main__":
    main()