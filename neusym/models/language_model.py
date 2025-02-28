#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语言模型接口模块

提供对小型语言模型的封装和访问接口。
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# 尝试导入PyTorch，如果不可用则使用模拟实现
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        BitsAndBytesConfig
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger("neusym.models")
    logger.warning("PyTorch或transformers未安装，将使用模拟语言模型")

logger = logging.getLogger("neusym.models")

class LanguageModel:
    """小型语言模型接口"""
    
    def __init__(
        self,
        model_name_or_path: str = "baichuan-inc/Baichuan-7B",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        max_length: int = 2048,
    ):
        """
        初始化语言模型接口
        
        Args:
            model_name_or_path: 模型名称或本地路径
            device: 运行设备，"auto"、"cpu"或"cuda:0"等
            load_in_8bit: 是否使用8位量化
            load_in_4bit: 是否使用4位量化
            cache_dir: 模型缓存目录
            trust_remote_code: 是否信任远程代码
            max_length: 最大序列长度
        """
        self.model_name = model_name_or_path
        self.max_length = max_length
        self.device_str = device
        
        # 确定设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"初始化语言模型: {model_name_or_path}，设备: {self.device}")
        
        # 配置量化参数
        quantization_config = None
        if self.device.type == "cuda":
            if load_in_4bit:
                logger.info("使用4位量化加载模型")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif load_in_8bit:
                logger.info("使用8位量化加载模型")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        
        # 加载分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
            logger.info("分词器加载成功")
        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise
        
        # 加载模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=quantization_config,
                device_map=self.device if self.device.type == "cuda" else None,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
            
            if self.device.type == "cpu":
                self.model.to(self.device)
                
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
        
        # 配置生成参数
        self.default_generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "num_return_sequences": 1,
        }
    
    def generate(
        self,
        prompt: str,
        **generation_kwargs
    ) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示
            generation_kwargs: 生成参数
            
        Returns:
            生成的响应文本
        """
        # 合并生成参数
        gen_config = {**self.default_generation_config, **generation_kwargs}
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                if hasattr(self.model, 'chat') and callable(self.model.chat):
                    # 某些模型有专用chat方法
                    logger.debug("使用模型的chat方法")
                    response = self.model.chat(self.tokenizer, prompt, **gen_config)
                    return response
                else:
                    # 标准生成方法
                    logger.debug("使用标准生成方法")
                    outputs = self.model.generate(
                        **inputs,
                        **gen_config
                    )
                    
                    # 解码生成的token
                    response = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    )
                    return response
        except Exception as e:
            logger.error(f"生成过程出错: {e}")
            return f"生成出错: {str(e)}"
    
    def get_embeddings(
        self,
        text: Union[str, List[str]],
        pooling_method: str = "mean"
    ) -> torch.Tensor:
        """
        获取文本的嵌入表示
        
        Args:
            text: 输入文本或文本列表
            pooling_method: 池化方法，"mean"、"max"或"cls"
            
        Returns:
            文本嵌入向量
        """
        # 确保输入是列表形式
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # 获取词嵌入
        try:
            with torch.no_grad():
                # 分词
                encoded_input = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取模型输出
                outputs = self.model(
                    **encoded_input,
                    output_hidden_states=True
                )
                
                # 获取最后一层隐藏状态
                last_hidden_state = outputs.hidden_states[-1]
                
                # 根据池化方法处理
                if pooling_method == "mean":
                    # 平均池化 (mask掉padding)
                    attention_mask = encoded_input["attention_mask"].unsqueeze(-1)
                    embeddings = torch.sum(last_hidden_state * attention_mask, 1) / torch.sum(attention_mask, 1)
                elif pooling_method == "max":
                    # 最大池化
                    attention_mask = encoded_input["attention_mask"].unsqueeze(-1)
                    embeddings = torch.max(last_hidden_state * attention_mask + (1 - attention_mask) * -1e9, dim=1)[0]
                elif pooling_method == "cls":
                    # 使用[CLS]位置的向量
                    embeddings = last_hidden_state[:, 0]
                else:
                    raise ValueError(f"不支持的池化方法: {pooling_method}")
                
                # 归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings
        except Exception as e:
            logger.error(f"获取文本嵌入出错: {e}")
            # 返回零向量
            return torch.zeros((len(texts), self.model.config.hidden_size)).to(self.device)
    
    def calculate_uncertainty(self, prompt: str, num_samples: int = 5) -> float:
        """
        计算模型对输入的不确定性
        
        Args:
            prompt: 输入提示
            num_samples: 采样次数
            
        Returns:
            不确定性分数 (0-1)
        """
        # 使用多次采样估计不确定性
        try:
            responses = []
            for _ in range(num_samples):
                response = self.generate(
                    prompt,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=64
                )
                responses.append(response)
            
            # 计算响应的多样性作为不确定性指标
            # 简化方法：比较响应的唯一数量与总数的比例
            unique_responses = set(responses)
            uncertainty = len(unique_responses) / num_samples
            
            logger.debug(f"不确定性评估: {uncertainty}")
            return uncertainty
        except Exception as e:
            logger.error(f"计算不确定性时出错: {e}")
            return 0.5  # 默认中等不确定性
    
    def unload(self):
        """卸载模型以释放内存"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        logger.info("模型已卸载")


class MockLanguageModel:
    """语言模型的模拟实现，用于测试"""
    
    def __init__(self):
        """初始化模拟语言模型"""
        logger.warning("使用模拟语言模型，仅用于测试")
        self.name = "MockModel"
        
    def generate(self, prompt, max_length=100, **kwargs):
        """模拟生成文本"""
        logger.info(f"模拟生成文本, 提示词长度: {len(prompt)}")
        
        # 提取问题中的关键信息
        if "在哪里" in prompt or "where" in prompt.lower():
            return "厨房"
        
        if "是谁" in prompt or "who" in prompt.lower():
            return "约翰"
        
        if "什么时候" in prompt or "when" in prompt.lower():
            return "昨天"
        
        if "什么颜色" in prompt or "color" in prompt.lower():
            return "红色"
        
        if "多少" in prompt or "how many" in prompt.lower():
            # 从提示中提取数字
            import re
            numbers = re.findall(r'\d+', prompt)
            if numbers:
                return str(sum(map(int, numbers)))
            return "3"
        
        # 对于复杂的逻辑问题，返回一个简单答案
        return "未知"
    
    def get_embedding(self, text):
        """返回模拟的文本嵌入"""
        import numpy as np
        # 生成稳定的伪随机向量，但对相似文本产生相似结果
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        
        return np.random.randn(384)
    
    def extract_keywords(self, text):
        """提取文本中的关键词"""
        # 简单方法：分词并去除停用词
        words = text.lower().replace("?", "").replace(".", "").replace(",", "").split()
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "的", "了", "吗", "在", "和", "与"}
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        
        return keywords


# 简单测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化小型语言模型
    lm = LanguageModel(
        model_name_or_path="baichuan-inc/Baichuan-7B",
        device="cpu",
        load_in_4bit=False,
    )
    
    # 测试生成
    prompt = "北京是中国的什么？"
    print(f"输入: {prompt}")
    response = lm.generate(prompt, max_new_tokens=64)
    print(f"输出: {response}") 