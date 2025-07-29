#!/bin/bash
# 模型下载脚本
# 使用huggingface-hub工具下载Qwen2.5系列模型到本地

echo "HDMARAG模型下载脚本"
echo "==================="

# 检查并安装huggingface-hub
echo "检查huggingface-hub工具..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "安装huggingface-hub..."
    pip install huggingface-hub[cli]
else
    echo "✓ huggingface-hub已安装"
fi

# 创建模型目录
echo "创建模型目录..."
mkdir -p models
cd models

# 检查可用磁盘空间
echo "检查磁盘空间..."
available_space=$(df . | tail -1 | awk '{print $4}')
echo "可用空间: $(($available_space / 1024 / 1024)) GB"

# 下载Qwen2.5-7B-Instruct (约14GB)
echo ""
echo "下载Qwen2.5-7B-Instruct (约14GB)..."
echo "这是推荐的入门模型，适合1张A100显卡"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir Qwen2.5-7B-Instruct \
    --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo "✓ Qwen2.5-7B-Instruct 下载完成"
else
    echo "✗ Qwen2.5-7B-Instruct 下载失败"
fi

# 询问是否下载更大的模型
echo ""
read -p "是否下载Qwen2.5-14B-Instruct? (约28GB, 需要2张A100) [y/N]: " download_14b
if [[ $download_14b =~ ^[Yy]$ ]]; then
    echo "下载Qwen2.5-14B-Instruct..."
    huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
        --local-dir Qwen2.5-14B-Instruct \
        --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✓ Qwen2.5-14B-Instruct 下载完成"
    else
        echo "✗ Qwen2.5-14B-Instruct 下载失败"
    fi
fi

echo ""
read -p "是否下载Qwen2.5-32B-Instruct? (约64GB, 需要4张A100) [y/N]: " download_32b
if [[ $download_32b =~ ^[Yy]$ ]]; then
    echo "下载Qwen2.5-32B-Instruct..."
    huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
        --local-dir Qwen2.5-32B-Instruct \
        --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✓ Qwen2.5-32B-Instruct 下载完成"
    else
        echo "✗ Qwen2.5-32B-Instruct 下载失败"
    fi
fi

# 可选：下载Qwen2.5-72B-Instruct (需要更多资源)
echo ""
read -p "是否下载Qwen2.5-72B-Instruct? (约145GB, 需要4张A100+) [y/N]: " download_72b
if [[ $download_72b =~ ^[Yy]$ ]]; then
    echo "下载Qwen2.5-72B-Instruct..."
    echo "警告: 此模型非常大，请确保有足够的磁盘空间和显存"
    huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
        --local-dir Qwen2.5-72B-Instruct \
        --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✓ Qwen2.5-72B-Instruct 下载完成"
    else
        echo "✗ Qwen2.5-72B-Instruct 下载失败"
    fi
fi

cd ..

# 显示下载结果
echo ""
echo "模型下载完成！"
echo "==============="
echo "已下载的模型:"
for model_dir in models/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        model_size=$(du -sh "$model_dir" | cut -f1)
        echo "  ✓ $model_name ($model_size)"
    fi
done

echo ""
echo "使用说明:"
echo "1. 运行测试: python test_hdmarag.py"
echo "2. 启动系统: python run_hdmarag.py --mode interactive"
echo "3. 快速测试: python run_hdmarag.py --mode quick"

# 检查GPU信息
echo ""
echo "GPU信息:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | nl
else
    echo "未检测到NVIDIA GPU或nvidia-smi命令"
fi

echo ""
echo "推荐配置:"
echo "  1张A100 (40GB) → 使用 Qwen2.5-7B-Instruct"
echo "  2张A100 (80GB) → 使用 Qwen2.5-14B-Instruct"
echo "  4张A100 (160GB) → 使用 Qwen2.5-32B-Instruct"
