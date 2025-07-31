import argparse
import asyncio
import json
import os
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from transformers import HfArgumentParser, logging
from vllm import SamplingParams

from hdmarag_system import HDMARAGSystem
from evaluation.longbench_utils import (
    DATASET2PROMPT, 
    DATASET2MAXNEWTOKENS,
    DATASET2CATEGORY,
    scorer,
    scorer_e
)

# 使用transformers的日志记录器
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# =====================================================================================
# 1. Command-line Arguments
# =====================================================================================
@dataclass
class Args:
    """评估脚本的参数"""
    model_name: str = field(
        metadata={"help": "要评估的本地模型的名称, 必须在local_model_config.json中定义。"}
    )
    datasets: List[str] = field(
        default_factory=lambda: ['hotpotqa', 'narrativeqa', 'qasper', 'gov_report'],
        metadata={"help": "要评估的数据集列表。"}
    )
    output_dir_root: str = field(
        default="results/longbench",
        metadata={"help": "保存所有评估结果的根目录。"}
    )
    eval_mode: str = field(
        default="baseline",
        metadata={"help": "评估模式: 'baseline' (仅模型) 或 'rag' (完整的HDMARAG系统)。"}
    )
    use_longbench_e: bool = field(
        default=False,
        metadata={"help": "是否在LongBench-E上进行评估（带有基于长度的评分）。"}
    )
    max_samples_per_dataset: int = field(
        default=20,
        metadata={"help": "每个数据集最多评估的样本数。-1表示全部。"}
    )

# =====================================================================================
# 2. Baseline Evaluation Function
# =====================================================================================
async def run_baseline_evaluation(hdmarag_system: HDMARAGSystem, args: Args, dataset_name: str, all_samples: List[Dict]) -> List[Dict]:
    """
    运行基线评估, 仅使用基础LLM而不使用RAG流程。
    """
    logger.info(f"[{dataset_name}] 评估模式: [Baseline] - 直接调用基础LLM。")
    
    model_wrapper = hdmarag_system.model_manager.models.get(args.model_name)
    if not model_wrapper or not hasattr(model_wrapper, 'llm'):
        logger.error(f"无法从模型管理器中获取已加载的LLM引擎 '{args.model_name}'。")
        return []
    llm = model_wrapper.llm

    predictions_with_metadata = []
    progress_bar = tqdm(all_samples, desc=f"Baseline Generating for {dataset_name}")

    for sample in progress_bar:
        prompt = DATASET2PROMPT[dataset_name]
        full_prompt = prompt.format(input=sample['input'], context=sample['context'])
        
        sampling_params = SamplingParams(temperature=0.01, top_p=1.0)
        sampling_params.max_tokens = DATASET2MAXNEWTOKENS.get(dataset_name, 2048)
        request_id = sample["_id"]

        try:
            final_output = None
            async for output in llm.generate(full_prompt, sampling_params, request_id):
                final_output = output
            
            prediction_text = final_output.outputs[0].text.strip() if final_output else ""
        except Exception as e:
            logger.error(f"vLLM生成错误 (ID: {request_id}): {e}")
            prediction_text = f"[GENERATION_ERROR]"
            
        # 复制样本并添加预测
        result_sample = sample.copy()
        result_sample['prediction'] = prediction_text
        predictions_with_metadata.append(result_sample)
        
    return predictions_with_metadata

# =====================================================================================
# 3. RAG Evaluation Function
# =====================================================================================
async def run_rag_evaluation(hdmarag_system: HDMARAGSystem, args: Args, dataset_name: str, all_samples: List[Dict]) -> List[Dict]:
    """
    运行完整的HDMARAG系统评估。
    """
    logger.info(f"[{dataset_name}] 评估模式: [HDMARAG] - 调用完整的RAG流程。")

    # 步骤 1: 为当前数据集构建知识库
    logger.info(f"[{dataset_name}] 正在构建或加载知识库...")
    try:
        await asyncio.to_thread(hdmarag_system.build_knowledge_base, dataset_name)
        logger.info(f"[{dataset_name}] 知识库准备就绪。")
    except Exception as e:
        logger.error(f"[{dataset_name}] 构建知识库失败: {e}", exc_info=True)
        return []

    predictions_with_metadata = []
    progress_bar = tqdm(all_samples, desc=f"HDMARAG Processing {dataset_name}")

    # 步骤 2: 循环处理样本
    for sample in progress_bar:
        try:
            rag_result = await asyncio.to_thread(hdmarag_system.process_sample, sample, dataset_name)
            prediction_text = rag_result.get("final_answer", "[RAG_ERROR: No answer found]")
        except Exception as e:
            logger.error(f"HDMARAG流程错误 (ID: {sample['_id']}): {e}", exc_info=True)
            prediction_text = f"[RAG_ERROR]"
            
        result_sample = sample.copy()
        result_sample['prediction'] = prediction_text
        predictions_with_metadata.append(result_sample)

    return predictions_with_metadata

# =====================================================================================
# 4. Main Execution
# =====================================================================================
async def main():
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]

    # 设置本次运行的专属输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir_root, f"{args.model_name}_{args.eval_mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"开始评估, 结果将保存在: {output_dir}")
    logger.info(f"参数: {args}")
    
    logger.info("正在初始化HDMARAG系统...")
    hdmarag_system = HDMARAGSystem()
    await hdmarag_system.initialize(model_name=args.model_name)
    logger.info("HDMARAG系统初始化完成。")

    overall_scores = {}
    
    for dataset_name in args.datasets:
        logger.info(f"===== 开始评估数据集: {dataset_name} =====")
        
        # 加载和准备数据集 (修正为从本地文件加载)
        try:
            import datasets
            local_data_path = os.path.join("data", f"{dataset_name}.jsonl")
            if not os.path.exists(local_data_path):
                logger.warning(f"本地数据文件未找到: {local_data_path}，跳过数据集 {dataset_name}。")
                continue
            raw_dataset = datasets.load_dataset('json', data_files={'test': local_data_path}, split='test')
            if args.max_samples_per_dataset != -1:
                raw_dataset = raw_dataset.select(range(min(args.max_samples_per_dataset, len(raw_dataset))))
            all_samples = list(raw_dataset)
        except Exception as e:
            logger.error(f"加载本地数据集 {dataset_name} 失败: {e}", exc_info=True)
            continue

        # 根据模式选择执行函数
        predictions_with_metadata = []
        if args.eval_mode == 'baseline':
            predictions_with_metadata = await run_baseline_evaluation(hdmarag_system, args, dataset_name, all_samples)
        elif args.eval_mode == 'rag':
            predictions_with_metadata = await run_rag_evaluation(hdmarag_system, args, dataset_name, all_samples)
        else:
            logger.error(f"未知的评估模式: '{args.eval_mode}'。程序退出。")
            return

        if not predictions_with_metadata:
            logger.warning(f"数据集 {dataset_name} 未产生任何预测。跳过评分。")
            continue
            
        # 提取用于评分的信息
        predictions = [item['prediction'] for item in predictions_with_metadata]
        answers = [item['answers'] for item in predictions_with_metadata]
        all_classes = predictions_with_metadata[0].get('all_classes', [])
        lengths = [item['length'] for item in predictions_with_metadata]

        # 计算分数 (修正了评分函数的调用)
        if args.use_longbench_e:
            score = scorer_e(dataset_name, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset_name, predictions, answers, all_classes)
        
        overall_scores[dataset_name] = score
        logger.info(f"数据集 [{dataset_name}] 的得分 ({args.eval_mode} 模式): {score}")

        # 保存该数据集的详细结果
        result_path = os.path.join(output_dir, f"{dataset_name}_predictions.jsonl")
        with open(result_path, "w", encoding="utf-8") as f:
            for item in predictions_with_metadata:
                res_to_save = {
                    "_id": item["_id"],
                    "prediction": item["prediction"],
                    "answers": item["answers"],
                }
                f.write(json.dumps(res_to_save, ensure_ascii=False) + "\n")
        logger.info(f"详细预测已保存至: {result_path}")

    logger.info("===== 评估任务总结 =====")
    logger.info(json.dumps(overall_scores, indent=2))

    # 保存总分总结
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_scores, f, indent=4)
    logger.info(f"评估完成。总结报告已保存至: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main()) 
