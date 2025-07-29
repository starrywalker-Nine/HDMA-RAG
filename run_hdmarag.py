#!/usr/bin/env python3
"""
HDMARAG主启动脚本
提供命令行界面来运行HDMARAG系统
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
from hdmarag_system import HDMARAGSystem

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    
    parser = argparse.ArgumentParser(
        description="HDMARAG (Hierarchical Declarative Memory Augment RAG) System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 交互式模式
  python run_hdmarag.py --mode interactive
  
  # 快速测试
  python run_hdmarag.py --mode quick --datasets multifieldqa_en hotpotqa --samples 5
  
  # 完整评估
  python run_hdmarag.py --mode full --datasets all --samples 50
  
  # 单个样本测试
  python run_hdmarag.py --mode single --question "What is AI?" --context "AI is..."
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["interactive", "quick", "full", "single", "benchmark"],
        default="interactive",
        help="运行模式"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["multifieldqa_en"],
        help="要评估的数据集列表，使用'all'表示所有数据集"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="每个数据集的最大样本数"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        help="单个样本模式下的问题"
    )
    
    parser.add_argument(
        "--context", 
        type=str,
        help="单个样本模式下的上下文"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-7b-instruct",
        help="使用的本地模型"
    )
    
    parser.add_argument(
        "--use-local",
        action="store_true",
        default=True,
        help="使用本地模型部署"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="hdmarag_config.json",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hdmarag_results",
        help="结果输出目录"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    return parser

def get_all_datasets() -> List[str]:
    """获取所有支持的数据集"""
    return [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
        "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
        "multi_news", "vcsum", "trec", "lsht", "samsum",
        "passage_retrieval_en", "passage_retrieval_zh", 
        "lcc", "repobench-p", "passage_count"
    ]

def interactive_mode(hdmarag_system: HDMARAGSystem):
    """交互式模式"""
    
    print("=" * 60)
    print("HDMARAG交互式模式")
    print("=" * 60)
    
    while True:
        print("\n请选择操作:")
        print("1. 处理单个问题")
        print("2. 评估数据集")
        print("3. 查看系统状态")
        print("4. 重置系统")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            handle_single_question(hdmarag_system)
        elif choice == "2":
            handle_dataset_evaluation(hdmarag_system)
        elif choice == "3":
            show_system_status(hdmarag_system)
        elif choice == "4":
            hdmarag_system.reset_system()
            print("系统已重置")
        elif choice == "5":
            print("退出HDMARAG系统")
            break
        else:
            print("无效选择，请重新输入")

def handle_single_question(hdmarag_system: HDMARAGSystem):
    """处理单个问题"""
    
    print("\n--- 单个问题处理 ---")
    
    question = input("请输入问题: ").strip()
    if not question:
        print("问题不能为空")
        return
    
    context = input("请输入上下文 (可选，按回车跳过): ").strip()
    if not context:
        context = "No specific context provided."
    
    print(f"\n处理问题: {question}")
    print("处理中...")
    
    sample = {
        "input": question,
        "context": context,
        "answers": [""]
    }
    
    result = hdmarag_system.process_sample(sample, "general")
    
    print(f"\n=== 处理结果 ===")
    print(f"问题: {result['question']}")
    print(f"答案: {result['final_answer']}")
    print(f"置信度: {result['confidence']}")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"成功: {'是' if result['success'] else '否'}")
    
    if not result['success'] and 'error' in result:
        print(f"错误: {result['error']}")

def handle_dataset_evaluation(hdmarag_system: HDMARAGSystem):
    """处理数据集评估"""
    
    print("\n--- 数据集评估 ---")
    
    # 显示可用数据集
    all_datasets = get_all_datasets()
    print("可用数据集:")
    for i, dataset in enumerate(all_datasets, 1):
        print(f"{i:2d}. {dataset}")
    
    # 选择数据集
    dataset_input = input("\n请输入数据集编号或名称 (多个用空格分隔): ").strip()
    
    selected_datasets = []
    for item in dataset_input.split():
        if item.isdigit():
            idx = int(item) - 1
            if 0 <= idx < len(all_datasets):
                selected_datasets.append(all_datasets[idx])
        elif item in all_datasets:
            selected_datasets.append(item)
    
    if not selected_datasets:
        print("未选择有效的数据集")
        return
    
    # 选择样本数量
    try:
        samples = int(input("请输入每个数据集的样本数量 (默认10): ").strip() or "10")
    except ValueError:
        samples = 10
    
    print(f"\n开始评估数据集: {selected_datasets}")
    print(f"每个数据集样本数: {samples}")
    
    try:
        results = hdmarag_system.evaluate_multiple_datasets(selected_datasets, samples)
        
        print(f"\n=== 评估完成 ===")
        overall_stats = results['overall_statistics']
        print(f"总样本数: {overall_stats['total_samples']}")
        print(f"成功样本数: {overall_stats['total_successful']}")
        print(f"成功率: {overall_stats['overall_success_rate']:.2%}")
        print(f"平均处理时间: {overall_stats['average_processing_time']:.2f}秒")
        
        # 保存结果
        hdmarag_system.save_results(results)
        
    except Exception as e:
        print(f"评估失败: {e}")

def show_system_status(hdmarag_system: HDMARAGSystem):
    """显示系统状态"""
    
    print("\n--- 系统状态 ---")
    
    status = hdmarag_system.get_system_status()
    
    print("性能指标:")
    metrics = status['performance_metrics']
    print(f"  总处理数: {metrics['total_processed']}")
    print(f"  平均处理时间: {metrics['average_processing_time']:.2f}秒")
    print(f"  平均增强分数: {metrics['average_enhancement_score']:.3f}")
    print(f"  记忆效率: {metrics['memory_efficiency']:.3f}")
    
    print("\n记忆摘要:")
    memory = status['memory_summary']
    total_memories = memory['total_memories']
    print(f"  情景记忆: {total_memories['episodic']}")
    print(f"  语义记忆: {total_memories['semantic']}")
    print(f"  程序记忆: {total_memories['procedural']}")
    print(f"  元记忆: {total_memories['meta']}")
    
    print(f"\n会话数: {status['session_count']}")
    if status['last_processed']:
        print(f"最后处理时间: {status['last_processed']}")

def quick_mode(hdmarag_system: HDMARAGSystem, datasets: List[str], samples: int):
    """快速测试模式"""
    
    print("=" * 60)
    print("HDMARAG快速测试模式")
    print("=" * 60)
    
    if "all" in datasets:
        datasets = get_all_datasets()[:3]  # 快速模式只测试前3个数据集
    
    print(f"测试数据集: {datasets}")
    print(f"每个数据集样本数: {samples}")
    
    try:
        results = hdmarag_system.evaluate_multiple_datasets(datasets, samples)
        
        print(f"\n=== 快速测试完成 ===")
        overall_stats = results['overall_statistics']
        print(f"总样本数: {overall_stats['total_samples']}")
        print(f"成功率: {overall_stats['overall_success_rate']:.2%}")
        print(f"平均处理时间: {overall_stats['average_processing_time']:.2f}秒")
        
        # 保存结果
        hdmarag_system.save_results(results)
        
        return True
        
    except Exception as e:
        print(f"快速测试失败: {e}")
        return False

def full_mode(hdmarag_system: HDMARAGSystem, datasets: List[str], samples: int):
    """完整评估模式"""
    
    print("=" * 60)
    print("HDMARAG完整评估模式")
    print("=" * 60)
    
    if "all" in datasets:
        datasets = get_all_datasets()
    
    print(f"评估数据集: {datasets}")
    print(f"每个数据集样本数: {samples}")
    print("这可能需要较长时间...")
    
    try:
        results = hdmarag_system.evaluate_multiple_datasets(datasets, samples)
        
        print(f"\n=== 完整评估完成 ===")
        overall_stats = results['overall_statistics']
        print(f"总样本数: {overall_stats['total_samples']}")
        print(f"成功率: {overall_stats['overall_success_rate']:.2%}")
        print(f"平均处理时间: {overall_stats['average_processing_time']:.2f}秒")
        
        # 详细统计
        print(f"\n=== 详细统计 ===")
        for dataset_name, dataset_result in results['evaluation_results'].items():
            stats = dataset_result.get('statistics', {})
            print(f"{dataset_name}:")
            print(f"  样本数: {dataset_result.get('processed_samples', 0)}")
            print(f"  成功率: {dataset_result.get('success_rate', 0):.2%}")
            print(f"  平均时间: {stats.get('average_processing_time', 0):.2f}秒")
        
        # 保存结果
        hdmarag_system.save_results(results)
        
        return True
        
    except Exception as e:
        print(f"完整评估失败: {e}")
        return False

def single_mode(hdmarag_system: HDMARAGSystem, question: str, context: str):
    """单个样本模式"""
    
    print("=" * 60)
    print("HDMARAG单个样本模式")
    print("=" * 60)
    
    if not question:
        print("错误: 必须提供问题")
        return False
    
    if not context:
        context = "No specific context provided."
    
    print(f"问题: {question}")
    print(f"上下文: {context[:100]}...")
    
    sample = {
        "input": question,
        "context": context,
        "answers": [""]
    }
    
    result = hdmarag_system.process_sample(sample, "general")
    
    print(f"\n=== 处理结果 ===")
    print(f"答案: {result['final_answer']}")
    print(f"置信度: {result['confidence']}")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"成功: {'是' if result['success'] else '否'}")
    
    if not result['success'] and 'error' in result:
        print(f"错误: {result['error']}")
    
    # 显示详细信息
    if result['success'] and 'hdmarag_result' in result:
        hdmarag_result = result['hdmarag_result']
        
        print(f"\n=== HDMARAG详细信息 ===")
        
        # 分层chunks信息
        hierarchical_chunks = hdmarag_result.get('hierarchical_chunks', {})
        if hierarchical_chunks:
            levels = hierarchical_chunks.get('levels', [])
            print(f"分层chunks: {len(levels)} 层")
            for i, level in enumerate(levels):
                print(f"  层级 {i+1}: {level.get('chunk_count', 0)} 个chunks")
        
        # Step-back insights
        step_back_insights = hdmarag_result.get('step_back_insights', {})
        if step_back_insights:
            print(f"Step-back抽象:")
            concept = step_back_insights.get('fundamental_concept', '')
            if concept:
                print(f"  核心概念: {concept}")
        
        # 性能指标
        metrics = hdmarag_result.get('performance_metrics', {})
        if metrics:
            print(f"性能指标:")
            print(f"  分层效率: {metrics.get('hierarchical_efficiency', 0):.3f}")
            print(f"  增强分数: {metrics.get('enhancement_score', 0):.3f}")
            print(f"  记忆利用率: {metrics.get('memory_utilization', 0):.3f}")
    
    return result['success']

def benchmark_mode(hdmarag_system: HDMARAGSystem):
    """基准测试模式"""
    
    print("=" * 60)
    print("HDMARAG基准测试模式")
    print("=" * 60)
    
    # 预定义的基准测试集
    benchmark_datasets = ["multifieldqa_en", "hotpotqa", "narrativeqa"]
    benchmark_samples = 20
    
    print(f"基准测试数据集: {benchmark_datasets}")
    print(f"每个数据集样本数: {benchmark_samples}")
    
    try:
        results = hdmarag_system.evaluate_multiple_datasets(benchmark_datasets, benchmark_samples)
        
        print(f"\n=== 基准测试完成 ===")
        overall_stats = results['overall_statistics']
        
        # 基准测试报告
        print(f"基准测试报告:")
        print(f"  总样本数: {overall_stats['total_samples']}")
        print(f"  成功率: {overall_stats['overall_success_rate']:.2%}")
        print(f"  平均处理时间: {overall_stats['average_processing_time']:.2f}秒")
        
        # 系统性能评估
        system_performance = results['system_performance']
        print(f"  平均增强分数: {system_performance['average_enhancement_score']:.3f}")
        print(f"  记忆效率: {system_performance['memory_efficiency']:.3f}")
        
        # 保存基准测试结果
        hdmarag_system.save_results(results, "hdmarag_benchmark_results")
        
        return True
        
    except Exception as e:
        print(f"基准测试失败: {e}")
        return False

def main():
    """主函数"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # 初始化HDMARAG系统
    try:
        hdmarag_system = HDMARAGSystem(
            model_name=args.model,
            config_path=args.config,
            use_local=args.use_local
        )
    except Exception as e:
        print(f"初始化HDMARAG系统失败: {e}")
        sys.exit(1)
    
    # 根据模式执行相应操作
    success = True
    
    if args.mode == "interactive":
        interactive_mode(hdmarag_system)
    
    elif args.mode == "quick":
        success = quick_mode(hdmarag_system, args.datasets, args.samples)
    
    elif args.mode == "full":
        success = full_mode(hdmarag_system, args.datasets, args.samples)
    
    elif args.mode == "single":
        success = single_mode(hdmarag_system, args.question, args.context)
    
    elif args.mode == "benchmark":
        success = benchmark_mode(hdmarag_system)
    
    # 退出状态
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()