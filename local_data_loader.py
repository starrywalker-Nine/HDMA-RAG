"""
本地LongBench数据加载器
从./longbench/data目录加载jsonl文件
"""

import json
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LocalLongBenchDataLoader:
    """本地LongBench数据加载器"""
    
    def __init__(self, data_dir: str = "./longbench/data"):
        self.data_dir = data_dir
        self.supported_datasets = [
            "narrativeqa",
            "qasper", 
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique"
        ]
        
        # 检查数据目录
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        logger.info(f"初始化本地数据加载器，数据目录: {data_dir}")
        self._check_available_datasets()
    
    def _check_available_datasets(self):
        """检查可用的数据集文件"""
        available_datasets = []
        missing_datasets = []
        
        for dataset in self.supported_datasets:
            file_path = os.path.join(self.data_dir, f"{dataset}.jsonl")
            if os.path.exists(file_path):
                available_datasets.append(dataset)
                file_size = os.path.getsize(file_path)
                logger.info(f"✓ 找到数据集: {dataset} ({file_size} bytes)")
            else:
                missing_datasets.append(dataset)
                logger.warning(f"✗ 缺失数据集: {dataset}")
        
        self.available_datasets = available_datasets
        self.missing_datasets = missing_datasets
        
        logger.info(f"可用数据集: {len(available_datasets)}/{len(self.supported_datasets)}")
        
        if missing_datasets:
            logger.warning(f"缺失数据集: {missing_datasets}")
    
    def load_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """加载指定数据集"""
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        if dataset_name not in self.available_datasets:
            raise FileNotFoundError(f"数据集文件不存在: {dataset_name}")
        
        file_path = os.path.join(self.data_dir, f"{dataset_name}.jsonl")
        
        logger.info(f"加载数据集: {dataset_name} from {file_path}")
        
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                        
                        # 限制样本数量
                        if max_samples and len(samples) >= max_samples:
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"解析JSON失败 {file_path}:{line_num}: {e}")
                        continue
            
            logger.info(f"成功加载 {len(samples)} 个样本 from {dataset_name}")
            return samples
            
        except Exception as e:
            logger.error(f"加载数据集失败 {dataset_name}: {e}")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        if dataset_name not in self.available_datasets:
            return {"error": f"数据集不可用: {dataset_name}"}
        
        file_path = os.path.join(self.data_dir, f"{dataset_name}.jsonl")
        
        # 统计样本数量
        sample_count = 0
        sample_preview = None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample_count += 1
                        if sample_preview is None:
                            try:
                                sample_preview = json.loads(line)
                            except:
                                pass
            
            file_size = os.path.getsize(file_path)
            
            return {
                "dataset_name": dataset_name,
                "file_path": file_path,
                "file_size": file_size,
                "sample_count": sample_count,
                "sample_preview": sample_preview
            }
            
        except Exception as e:
            return {"error": f"获取数据集信息失败: {e}"}
    
    def list_available_datasets(self) -> List[str]:
        """列出所有可用的数据集"""
        return self.available_datasets.copy()
    
    def validate_dataset_format(self, dataset_name: str) -> Dict[str, Any]:
        """验证数据集格式"""
        if dataset_name not in self.available_datasets:
            return {"valid": False, "error": f"数据集不可用: {dataset_name}"}
        
        file_path = os.path.join(self.data_dir, f"{dataset_name}.jsonl")
        
        required_fields = ["input", "context", "answers"]
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sample_count": 0,
            "field_coverage": {}
        }
        
        field_counts = {field: 0 for field in required_fields}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    validation_result["sample_count"] += 1
                    
                    try:
                        sample = json.loads(line)
                        
                        # 检查必需字段
                        for field in required_fields:
                            if field in sample:
                                field_counts[field] += 1
                            else:
                                validation_result["errors"].append(
                                    f"行 {line_num}: 缺失字段 '{field}'"
                                )
                        
                        # 检查answers字段格式
                        if "answers" in sample:
                            if not isinstance(sample["answers"], list):
                                validation_result["warnings"].append(
                                    f"行 {line_num}: 'answers' 应该是列表格式"
                                )
                            elif len(sample["answers"]) == 0:
                                validation_result["warnings"].append(
                                    f"行 {line_num}: 'answers' 列表为空"
                                )
                        
                        # 只检查前100个样本以提高速度
                        if line_num >= 100:
                            break
                            
                    except json.JSONDecodeError as e:
                        validation_result["errors"].append(
                            f"行 {line_num}: JSON解析错误: {e}"
                        )
            
            # 计算字段覆盖率
            for field in required_fields:
                coverage = field_counts[field] / max(validation_result["sample_count"], 1)
                validation_result["field_coverage"][field] = coverage
                
                if coverage < 0.9:  # 如果覆盖率低于90%
                    validation_result["warnings"].append(
                        f"字段 '{field}' 覆盖率较低: {coverage:.1%}"
                    )
            
            # 如果有错误，标记为无效
            if validation_result["errors"]:
                validation_result["valid"] = False
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"验证数据集格式失败: {e}"
            }

def test_local_data_loader():
    """测试本地数据加载器"""
    print("测试本地数据加载器")
    print("=" * 50)
    
    try:
        # 创建数据加载器
        loader = LocalLongBenchDataLoader()
        
        # 列出可用数据集
        available = loader.list_available_datasets()
        print(f"可用数据集: {available}")
        
        # 测试加载每个数据集
        for dataset_name in available:
            print(f"\n测试数据集: {dataset_name}")
            
            # 获取数据集信息
            info = loader.get_dataset_info(dataset_name)
            print(f"  样本数量: {info.get('sample_count', 'unknown')}")
            print(f"  文件大小: {info.get('file_size', 0)} bytes")
            
            # 验证格式
            validation = loader.validate_dataset_format(dataset_name)
            print(f"  格式有效: {validation['valid']}")
            if validation.get('errors'):
                print(f"  错误: {len(validation['errors'])} 个")
            if validation.get('warnings'):
                print(f"  警告: {len(validation['warnings'])} 个")
            
            # 加载少量样本测试
            try:
                samples = loader.load_dataset(dataset_name, max_samples=2)
                print(f"  成功加载: {len(samples)} 个样本")
                
                if samples:
                    sample = samples[0]
                    print(f"  样本字段: {list(sample.keys())}")
                    print(f"  问题长度: {len(sample.get('input', ''))}")
                    print(f"  上下文长度: {len(sample.get('context', ''))}")
                    
            except Exception as e:
                print(f"  加载失败: {e}")
        
        print(f"\n测试完成!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    test_local_data_loader()