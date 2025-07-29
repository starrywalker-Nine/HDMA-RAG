#!/usr/bin/env python3
"""
项目清理脚本
移除无关文件，保留HDMARAG核心组件
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """清理项目，移除无关文件"""
    
    print("开始清理HDMARAG项目...")
    
    # 要保留的核心文件
    keep_files = {
        # HDMARAG核心文件
        "hdmarag_core.py",
        "hdmarag_system.py", 
        "hdmarag_config.json",
        "local_model_interface.py",
        "local_model_config.json",
        "run_hdmarag.py",
        "test_hdmarag.py",
        
        # 文档文件
        "HDMARAG_README.md",
        "PROJECT_RESTRUCTURE.md",
        "requirements.txt",
        
        # 原有组件（作为备用）
        "testQA.py",
        "testChunks.py", 
        "testRetriever.py",
        "answerModel.py",
        
        # 数据加载器（可能需要）
        "local_data_loader.py",
        
        # 配置文件
        "cleanup_project.py"
    }
    
    # 要移除的文件
    remove_files = {
        "gpu_server_rag_system_v2.py",  # 旧的GPU服务器文件
        "local_longbench_evaluation_v2.py",  # 旧的评估文件
        "local_rag_baselines.py",  # 旧的基线方法
        "run_local_evaluation.py",  # 旧的评估脚本
        "gpu_server_config.json",  # 旧的GPU配置
        "README.md",  # 旧的README，用新的替代
        ".DS_Store"  # macOS系统文件
    }
    
    # 移除无关文件
    for file_name in remove_files:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                print(f"✓ 已移除: {file_name}")
            except Exception as e:
                print(f"✗ 移除失败 {file_name}: {e}")
    
    # 移除__pycache__目录
    if os.path.exists("__pycache__"):
        try:
            shutil.rmtree("__pycache__")
            print("✓ 已移除: __pycache__")
        except Exception as e:
            print(f"✗ 移除__pycache__失败: {e}")
    
    # 创建新的目录结构
    directories = [
        "models",  # 存放本地模型
        "data",    # 存放数据集
        "results", # 存放结果
        "logs",    # 存放日志
        "configs"  # 存放配置文件
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"✓ 已创建目录: {directory}")
            except Exception as e:
                print(f"✗ 创建目录失败 {directory}: {e}")
    
    # 移动配置文件到configs目录
    config_files = ["hdmarag_config.json", "local_model_config.json"]
    for config_file in config_files:
        if os.path.exists(config_file):
            target_path = os.path.join("configs", config_file)
            if not os.path.exists(target_path):
                try:
                    shutil.copy2(config_file, target_path)
                    print(f"✓ 已复制配置文件到: {target_path}")
                except Exception as e:
                    print(f"✗ 复制配置文件失败: {e}")
    
    # 创建模型下载脚本
    create_model_download_script()
    
    # 创建快速启动脚本
    create_quick_start_script()
    
    print("\n项目清理完成！")
    print("\n当前项目结构:")
    print_project_structure()

def create_model_download_script():
    """创建模型下载脚本"""
    
    script_content = '''#!/bin/bash
# 模型下载脚本
# 下载Qwen2.5系列模型到本地

echo "开始下载Qwen2.5模型..."

# 创建模型目录
mkdir -p models

# 下载Qwen2.5-7B-Instruct
echo "下载Qwen2.5-7B-Instruct..."
cd models
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

# 下载Qwen2.5-14B-Instruct (可选)
echo "下载Qwen2.5-14B-Instruct..."
git clone https://huggingface.co/Qwen/Qwen2.5-14B-Instruct

# 下载Qwen2.5-32B-Instruct (可选，需要更多显存)
# echo "下载Qwen2.5-32B-Instruct..."
# git clone https://huggingface.co/Qwen/Qwen2.5-32B-Instruct

cd ..
echo "模型下载完成！"
'''
    
    with open("download_models.sh", "w") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod("download_models.sh", 0o755)
    print("✓ 已创建模型下载脚本: download_models.sh")

def create_quick_start_script():
    """创建快速启动脚本"""
    
    script_content = '''#!/bin/bash
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
'''
    
    with open("quick_start.sh", "w") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod("quick_start.sh", 0o755)
    print("✓ 已创建快速启动脚本: quick_start.sh")

def print_project_structure():
    """打印项目结构"""
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = sorted(os.listdir(directory))
        for i, item in enumerate(items):
            if item.startswith('.'):
                continue
                
            path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item}")
            
            if os.path.isdir(path) and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(path, next_prefix, max_depth, current_depth + 1)
    
    print("HDMARAG/")
    print_tree(".", max_depth=2)

if __name__ == "__main__":
    cleanup_project()