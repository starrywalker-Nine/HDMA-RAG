#!/bin/bash
# HDMARAG快速启动脚本

echo "HDMARAG系统快速启动"
echo "===================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -o "3\.[0-9]*")
echo "Python版本: $python_version"

# 检查GPU
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 检查模型文件
echo "检查模型文件..."
if [ -d "models/Qwen2.5-7B-Instruct" ]; then
    echo "✓ 找到Qwen2.5-7B-Instruct模型"
else
    echo "✗ 未找到模型文件，请运行 ./download_models.sh"
    exit 1
fi

# 运行测试
echo "运行系统测试..."
python3 test_hdmarag.py

# 启动交互式模式
echo "启动HDMARAG交互式模式..."
python3 run_hdmarag.py --mode interactive
