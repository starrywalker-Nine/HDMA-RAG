#!/bin/bash
# HDMARAG LongBench 评估启动脚本

echo "========================================="
echo "  HDMARAG LongBench Evaluation Launcher"
echo "========================================="

# 激活您的conda环境 (如果需要，请修改环境名称)
# source /opt/conda/bin/activate your_env_name

# --- 配置 ---
# 要评估的模型 (必须是 local_model_config.json 中定义过的)
MODEL_NAME="qwen2.5-7b-instruct"

# 要评估的数据集 (all, 或者用空格隔开的列表, e.g., "hotpotqa narrativeqa")
# LongBench all datasets: narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news multifieldqa_zh dureader vcsum trec triviaqa samsum lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p
DATASETS_TO_EVAL=("hotpotqa" "narrativeqa" "qasper" "gov_report")

# 每个数据集评估的样本数 (-1 代表全部)
MAX_SAMPLES=20

# 评估结果输出目录
OUTPUT_DIR="results/longbench_eval_$(date +%Y%m%d_%H%M%S)"

# 是否使用 LongBench-E (按长度评估)
# 设置为 true 来启用
USE_LONGTERM_E=false 

# --- 执行 ---
echo "Starting evaluation with the following settings:"
echo "  - Model: $MODEL_NAME"
echo "  - Datasets: ${DATASETS_TO_EVAL[*]}"
echo "  - Samples/Dataset: $MAX_SAMPLES"
echo "  - Output Directory: $OUTPUT_DIR"
echo "  - LongBench-E Mode: $USE_LONGTERM_E"
echo "-----------------------------------------"

# 构建命令
# 注意：不再使用 torchrun，而是直接用 python
COMMAND="python evaluation/longbench_eval.py \
    --model_name $MODEL_NAME \
    --dataset_names ${DATASETS_TO_EVAL[*]} \
    --max_samples_per_dataset $MAX_SAMPLES \
    --output_dir $OUTPUT_DIR"

if [ "$USE_LONGTERM_E" = true ]; then
    COMMAND="$COMMAND --use_longbench_e"
fi

# 运行命令
eval $COMMAND

echo "========================================="
echo "Evaluation finished."
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================" 
