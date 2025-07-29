"""
本地模型接口 - 支持vLLM加速和A100显卡
专为超长上下文处理优化
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# 检查vLLM是否可用
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    
    # 尝试导入destroy_model_parallel，如果失败则使用备用方案
    try:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        DESTROY_MODEL_PARALLEL_AVAILABLE = True
    except ImportError:
        try:
            # 尝试新的导入路径
            from vllm.distributed.parallel_state import destroy_model_parallel
            DESTROY_MODEL_PARALLEL_AVAILABLE = True
        except ImportError:
            # 如果都失败，使用备用方案
            DESTROY_MODEL_PARALLEL_AVAILABLE = False
            def destroy_model_parallel():
                """备用的模型并行销毁函数"""
                pass
                
except ImportError:
    VLLM_AVAILABLE = False
    DESTROY_MODEL_PARALLEL_AVAILABLE = False
    print("警告: vLLM未安装，将使用transformers作为备用")
    
    def destroy_model_parallel():
        """备用的模型并行销毁函数"""
        pass

# 备用transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str
    model_path: str
    max_model_len: int = 131072  # 128K上下文
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    enforce_eager: bool = False
    max_num_seqs: int = 256

class LocalModelManager:
    """本地模型管理器 - 支持vLLM和多GPU"""
    
    def __init__(self, config_path: str = "local_model_config.json"):
        self.config_path = config_path
        self.models = {}
        
        # 设置日志 - 需要在load_config之前初始化
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.gpu_info = self._get_gpu_info()
        self.load_config()
        
    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        if not torch.cuda.is_available():
            return {"gpu_count": 0, "gpu_memory": []}
        
        gpu_count = torch.cuda.device_count()
        gpu_memory = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            gpu_memory.append({
                "device_id": i,
                "name": props.name,
                "memory_gb": round(memory_gb, 1),
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return {
            "gpu_count": gpu_count,
            "gpu_memory": gpu_memory,
            "total_memory_gb": sum(gpu["memory_gb"] for gpu in gpu_memory)
        }
    
    def load_config(self):
        """加载模型配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            config_data = self._get_default_config()
            self.save_config(config_data)
        
        self.model_configs = {}
        for model_name, model_data in config_data.get("models", {}).items():
            self.model_configs[model_name] = ModelConfig(**model_data)
        
        self.logger.info(f"加载了 {len(self.model_configs)} 个模型配置")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "models": {
                "qwen2.5-7b-instruct": {
                    "model_name": "qwen2.5-7b-instruct",
                    "model_path": "./models/Qwen2.5-7B-Instruct",
                    "max_model_len": 131072,
                    "gpu_memory_utilization": 0.9,
                    "tensor_parallel_size": 1,
                    "dtype": "bfloat16"
                },
                "qwen2.5-14b-instruct": {
                    "model_name": "qwen2.5-14b-instruct", 
                    "model_path": "./models/Qwen2.5-14B-Instruct",
                    "max_model_len": 131072,
                    "gpu_memory_utilization": 0.9,
                    "tensor_parallel_size": 2,
                    "dtype": "bfloat16"
                },
                "qwen2.5-32b-instruct": {
                    "model_name": "qwen2.5-32b-instruct",
                    "model_path": "./models/Qwen2.5-32B-Instruct", 
                    "max_model_len": 131072,
                    "gpu_memory_utilization": 0.9,
                    "tensor_parallel_size": 4,
                    "dtype": "bfloat16"
                }
            },
            "gpu_allocation": {
                "auto_detect": True,
                "preferred_models": ["qwen2.5-7b-instruct", "qwen2.5-14b-instruct"],
                "fallback_model": "qwen2.5-7b-instruct"
            }
        }
    
    def save_config(self, config_data: Dict[str, Any]):
        """保存配置"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    def get_optimal_model_config(self, preferred_model: str = None) -> ModelConfig:
        """根据GPU资源获取最优模型配置"""
        available_memory = self.gpu_info["total_memory_gb"]
        gpu_count = self.gpu_info["gpu_count"]
        
        self.logger.info(f"可用GPU: {gpu_count}张, 总显存: {available_memory:.1f}GB")
        
        # 如果指定了首选模型且可用，优先使用
        if preferred_model and preferred_model in self.model_configs:
            config = self.model_configs[preferred_model]
            if self._check_model_feasible(config):
                return config
        
        # 根据显存自动选择模型
        if available_memory >= 120 and gpu_count >= 4:  # 4张A100
            model_name = "qwen2.5-32b-instruct"
        elif available_memory >= 80 and gpu_count >= 2:  # 2张A100
            model_name = "qwen2.5-14b-instruct"
        else:  # 1张A100
            model_name = "qwen2.5-7b-instruct"
        
        if model_name in self.model_configs:
            config = self.model_configs[model_name]
            # 调整tensor_parallel_size
            config.tensor_parallel_size = min(config.tensor_parallel_size, gpu_count)
            return config
        
        # 备用配置
        return ModelConfig(
            model_name="qwen2.5-7b-instruct",
            model_path="./models/Qwen2.5-7B-Instruct",
            max_model_len=131072,
            tensor_parallel_size=1
        )
    
    def _check_model_feasible(self, config: ModelConfig) -> bool:
        """检查模型是否可行"""
        required_gpus = config.tensor_parallel_size
        available_gpus = self.gpu_info["gpu_count"]
        
        if required_gpus > available_gpus:
            return False
        
        # 简单的显存估算
        if "7b" in config.model_name.lower():
            required_memory_per_gpu = 20
        elif "14b" in config.model_name.lower():
            required_memory_per_gpu = 35
        elif "32b" in config.model_name.lower():
            required_memory_per_gpu = 35
        else:
            required_memory_per_gpu = 20
        
        available_memory_per_gpu = self.gpu_info["total_memory_gb"] / available_gpus
        
        return available_memory_per_gpu >= required_memory_per_gpu
    
    def load_model(self, model_name: str = None) -> 'LocalModel':
        """加载模型"""
        if model_name and model_name in self.models:
            return self.models[model_name]
        
        config = self.get_optimal_model_config(model_name)
        
        self.logger.info(f"加载模型: {config.model_name}")
        self.logger.info(f"模型路径: {config.model_path}")
        self.logger.info(f"最大长度: {config.max_model_len}")
        self.logger.info(f"并行度: {config.tensor_parallel_size}")
        
        if VLLM_AVAILABLE:
            model = VLLMModel(config, self.logger)
        elif TRANSFORMERS_AVAILABLE:
            model = TransformersModel(config, self.logger)
        else:
            raise RuntimeError("没有可用的模型后端 (vLLM或transformers)")
        
        self.models[config.model_name] = model
        return model
    
    def unload_model(self, model_name: str):
        """卸载模型"""
        if model_name in self.models:
            self.models[model_name].unload()
            del self.models[model_name]
            self.logger.info(f"已卸载模型: {model_name}")

class LocalModel:
    """本地模型基类"""
    
    def __init__(self, config: ModelConfig, logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.tokenizer = None
        
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1, 
                 top_p: float = 0.9, stop: List[str] = None) -> str:
        """生成文本"""
        raise NotImplementedError
    
    def batch_generate(self, prompts: List[str], max_tokens: int = 2048, 
                      temperature: float = 0.1, top_p: float = 0.9) -> List[str]:
        """批量生成"""
        raise NotImplementedError
    
    def unload(self):
        """卸载模型"""
        raise NotImplementedError

class VLLMModel(LocalModel):
    """vLLM模型实现"""
    
    def __init__(self, config: ModelConfig, logger):
        super().__init__(config, logger)
        self.load_model()
    
    def load_model(self):
        """加载vLLM模型"""
        try:
            # 设置环境变量
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.config.tensor_parallel_size))
            
            self.model = LLM(
                model=self.config.model_path,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
                enforce_eager=self.config.enforce_eager,
                max_num_seqs=self.config.max_num_seqs
            )
            
            self.logger.info(f"vLLM模型加载成功: {self.config.model_name}")
            
        except Exception as e:
            self.logger.error(f"vLLM模型加载失败: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1,
                 top_p: float = 0.9, stop: List[str] = None) -> str:
        """生成文本"""
        try:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or []
            )
            
            outputs = self.model.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"文本生成失败: {e}")
            return f"生成失败: {str(e)}"
    
    def batch_generate(self, prompts: List[str], max_tokens: int = 2048,
                      temperature: float = 0.1, top_p: float = 0.9) -> List[str]:
        """批量生成"""
        try:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            outputs = self.model.generate(prompts, sampling_params)
            return [output.outputs[0].text.strip() for output in outputs]
            
        except Exception as e:
            self.logger.error(f"批量生成失败: {e}")
            return [f"生成失败: {str(e)}"] * len(prompts)
    
    def unload(self):
        """卸载模型"""
        if self.model:
            try:
                # 尝试销毁模型并行状态
                if DESTROY_MODEL_PARALLEL_AVAILABLE:
                    destroy_model_parallel()
                
                del self.model
                self.model = None
                torch.cuda.empty_cache()
                self.logger.info("vLLM模型已卸载")
            except Exception as e:
                self.logger.error(f"模型卸载失败: {e}")
                # 强制清理
                try:
                    del self.model
                    self.model = None
                    torch.cuda.empty_cache()
                except:
                    pass

class TransformersModel(LocalModel):
    """Transformers模型实现（备用）"""
    
    def __init__(self, config: ModelConfig, logger):
        super().__init__(config, logger)
        self.load_model()
    
    def load_model(self):
        """加载transformers模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16,
                device_map="auto",
                trust_remote_code=self.config.trust_remote_code
            )
            
            self.logger.info(f"Transformers模型加载成功: {self.config.model_name}")
            
        except Exception as e:
            self.logger.error(f"Transformers模型加载失败: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1,
                 top_p: float = 0.9, stop: List[str] = None) -> str:
        """生成文本"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"文本生成失败: {e}")
            return f"生成失败: {str(e)}"
    
    def batch_generate(self, prompts: List[str], max_tokens: int = 2048,
                      temperature: float = 0.1, top_p: float = 0.9) -> List[str]:
        """批量生成"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens, temperature, top_p)
            results.append(result)
        return results
    
    def unload(self):
        """卸载模型"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        self.logger.info("Transformers模型已卸载")

# 全局模型管理器实例
_model_manager = None

def get_model_manager() -> LocalModelManager:
    """获取全局模型管理器"""
    global _model_manager
    if _model_manager is None:
        _model_manager = LocalModelManager()
    return _model_manager

def load_local_model(model_name: str = None) -> LocalModel:
    """加载本地模型"""
    manager = get_model_manager()
    return manager.load_model(model_name)

def main():
    """测试本地模型接口"""
    print("=" * 50)
    print("本地模型接口测试")
    print("=" * 50)
    
    # 初始化模型管理器
    manager = LocalModelManager()
    
    # 显示GPU信息
    print(f"GPU信息: {manager.gpu_info}")
    
    # 获取最优配置
    config = manager.get_optimal_model_config()
    print(f"推荐模型: {config.model_name}")
    print(f"并行度: {config.tensor_parallel_size}")
    print(f"最大长度: {config.max_model_len}")
    
    # 测试模型加载（如果模型文件存在）
    try:
        model = manager.load_model()
        print("模型加载成功")
        
        # 测试生成
        test_prompt = "请简要介绍人工智能的发展历程。"
        response = model.generate(test_prompt, max_tokens=200)
        print(f"测试生成: {response}")
        
        # 卸载模型
        manager.unload_model(config.model_name)
        print("模型卸载成功")
        
    except Exception as e:
        print(f"模型测试失败: {e}")
        print("请确保模型文件存在于指定路径")

if __name__ == "__main__":
    main()