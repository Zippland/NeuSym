#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经-符号接口模块

实现神经网络和符号系统之间的双向转换，包括自然语言到逻辑形式的转换，
以及逻辑推理结果到自然语言的转换。
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any

from .symbolic_engine import Predicate, Fact

logger = logging.getLogger("neusym.interface")

class NeuralSymbolicInterface:
    """神经-符号接口，实现两个系统之间的转换"""
    
    def __init__(self, language_model):
        """
        初始化神经-符号接口
        
        Args:
            language_model: 语言模型实例
        """
        self.language_model = language_model
        self.pattern_matchers = {}
        self.templates = {}
        
        # 初始化常用的模式匹配器
        self._init_pattern_matchers()
        
        # 初始化自然语言生成模板
        self._init_templates()
        
        logger.info("神经-符号接口初始化完成")
    
    def _init_pattern_matchers(self):
        """初始化常用的模式匹配器"""
        # 位置关系模式
        self.pattern_matchers["location"] = {
            "pattern": r"(\w+)(?:在|位于|去了)(\w+)",
            "predicate": "位于",
            "args_order": [0, 1]  # 参数顺序：[实体, 位置]
        }
        
        # 所有权关系模式
        self.pattern_matchers["possession"] = {
            "pattern": r"(\w+)(?:拥有|有|持有)(\w+)",
            "predicate": "拥有",
            "args_order": [0, 1]  # 参数顺序：[拥有者, 物品]
        }
        
        # 给予关系模式
        self.pattern_matchers["giving"] = {
            "pattern": r"(\w+)(?:给了|给|赠送)(\w+)(?:给|到)(\w+)",
            "predicate": "给予",
            "args_order": [0, 2, 1]  # 参数顺序：[给予者, 接收者, 物品]
        }
        
        # 否定位置关系
        self.pattern_matchers["negative_location"] = {
            "pattern": r"(\w+)(?:不在|不位于|没去)(\w+)",
            "predicate": "位于",
            "args_order": [0, 1],  # 参数顺序：[实体, 位置]
            "negated": True
        }
        
        # 左右位置关系
        self.pattern_matchers["spatial_left"] = {
            "pattern": r"(\w+)(?:在|位于)(\w+)的左边",
            "predicate": "左侧",
            "args_order": [0, 1]  # 参数顺序：[实体, 参照物]
        }
        
        self.pattern_matchers["spatial_right"] = {
            "pattern": r"(\w+)(?:在|位于)(\w+)的右边",
            "predicate": "右侧",
            "args_order": [0, 1]  # 参数顺序：[实体, 参照物]
        }
        
        # 颜色属性
        self.pattern_matchers["color"] = {
            "pattern": r"(\w+)是(\w+)色的",
            "predicate": "颜色",
            "args_order": [0, 1]  # 参数顺序：[实体, 颜色]
        }
        
        # 大小关系
        self.pattern_matchers["size_compare"] = {
            "pattern": r"(\w+)比(\w+)(\w+)",
            "predicate": "比较",
            "args_order": [0, 1, 2]  # 参数顺序：[实体1, 实体2, 关系]
        }
    
    def _init_templates(self):
        """初始化自然语言生成模板"""
        # 位置关系模板
        self.templates["位于"] = "{0}在{1}"
        self.templates["位于_否定"] = "{0}不在{1}"
        
        # 所有权关系模板
        self.templates["拥有"] = "{0}拥有{1}"
        self.templates["拥有_否定"] = "{0}没有{1}"
        
        # 给予关系模板
        self.templates["给予"] = "{0}给了{2}给{1}"
        
        # 空间关系模板
        self.templates["左侧"] = "{0}在{1}的左边"
        self.templates["右侧"] = "{0}在{1}的右边"
        
        # 颜色属性模板
        self.templates["颜色"] = "{0}是{1}色的"
        
        # 大小关系模板
        self.templates["大于"] = "{0}比{1}大"
        self.templates["小于"] = "{0}比{1}小"
        
        # 通用模板
        self.templates["default"] = "{关系}({参数})"
    
    def text_to_logical_form(self, text: str) -> List[Fact]:
        """
        将自然语言文本转换为逻辑形式
        
        Args:
            text: 输入的自然语言文本
            
        Returns:
            转换后的逻辑事实列表
        """
        logger.debug(f"转换文本到逻辑形式: {text}")
        facts = []
        
        # 使用模式匹配器尝试匹配
        for matcher_name, matcher in self.pattern_matchers.items():
            pattern = matcher["pattern"]
            
            for match in re.finditer(pattern, text):
                # 提取匹配的参数
                args = list(match.groups())
                
                # 按指定顺序重排参数
                ordered_args = [args[i] for i in matcher["args_order"]]
                
                # 创建谓词和事实
                predicate = Predicate(matcher["predicate"], ordered_args)
                is_negated = matcher.get("negated", False)
                fact = Fact(predicate, is_negated)
                
                facts.append(fact)
                logger.debug(f"模式{matcher_name}匹配: {fact}")
        
        # 如果没有匹配到任何模式，尝试使用语言模型进行分析
        if not facts and hasattr(self.language_model, "generate"):
            try:
                # 构造提示词请求语言模型进行分析
                prompt = f"""
                请将以下自然语言文本转换为逻辑形式，使用谓词逻辑表示:
                
                文本: "{text}"
                
                使用以下格式输出(不要添加其他解释):
                谓词(参数1, 参数2, ...)
                """
                
                response = self.language_model.generate(prompt, max_new_tokens=50)
                
                # 解析语言模型输出
                # 简化处理：查找形如 predicate(arg1, arg2) 的模式
                logic_pattern = r"(\w+)\(([^)]+)\)"
                for match in re.finditer(logic_pattern, response):
                    pred_name = match.group(1)
                    args_str = match.group(2)
                    args = [arg.strip() for arg in args_str.split(",")]
                    
                    # 创建谓词和事实
                    predicate = Predicate(pred_name, args)
                    fact = Fact(predicate)
                    facts.append(fact)
                    logger.debug(f"语言模型分析: {fact}")
            
            except Exception as e:
                logger.error(f"使用语言模型分析文本时出错: {e}")
        
        # 如果仍未提取到任何事实，创建一个包含完整文本的通用事实
        if not facts:
            predicate = Predicate("陈述", [text])
            fact = Fact(predicate)
            facts.append(fact)
            logger.debug(f"创建通用事实: {fact}")
        
        return facts
    
    def logical_form_to_text(self, fact: Fact) -> str:
        """
        将逻辑形式转换为自然语言文本
        
        Args:
            fact: 逻辑事实
            
        Returns:
            转换后的自然语言文本
        """
        logger.debug(f"转换逻辑形式到文本: {fact}")
        
        predicate = fact.predicate
        pred_name = predicate.name
        args = predicate.args
        
        # 检查是否有对应的模板
        template_key = pred_name
        if fact.is_negated:
            template_key += "_否定"
        
        if template_key in self.templates:
            # 使用模板生成文本
            template = self.templates[template_key]
            text = template.format(*args)
            logger.debug(f"使用模板生成: {text}")
            return text
        
        # 没有对应的模板，使用默认格式
        args_str = ", ".join(args)
        if fact.is_negated:
            text = f"不是{pred_name}({args_str})"
        else:
            text = f"{pred_name}({args_str})"
        
        logger.debug(f"使用默认格式生成: {text}")
        return text
    
    def generate_explanation(self, facts: List[Fact], question: str) -> str:
        """
        基于事实和问题生成解释
        
        Args:
            facts: 相关事实列表
            question: 原始问题
            
        Returns:
            生成的解释文本
        """
        # 将事实转换为自然语言
        fact_texts = [self.logical_form_to_text(fact) for fact in facts]
        
        # 如果没有事实，返回简单回复
        if not facts:
            return "我没有找到相关的信息来回答这个问题。"
        
        # 使用语言模型生成连贯的解释
        if hasattr(self.language_model, "generate"):
            try:
                # 构造提示词
                prompt = f"""
                基于以下事实，生成对问题的解释:
                
                问题: {question}
                
                已知事实:
                {". ".join(fact_texts)}
                
                请生成清晰、简洁的解释:
                """
                
                explanation = self.language_model.generate(prompt, max_new_tokens=100)
                return explanation.strip()
                
            except Exception as e:
                logger.error(f"生成解释时出错: {e}")
        
        # 语言模型不可用或出错，使用简单拼接
        return f"根据我所知: {'. '.join(fact_texts)}"
    
    def extract_query_predicate(self, question: str) -> Tuple[Optional[Predicate], bool]:
        """
        从问题中提取查询谓词
        
        Args:
            question: 问题文本
            
        Returns:
            (查询谓词, 是否为否定查询)的元组，如果无法提取则谓词为None
        """
        logger.debug(f"从问题提取查询谓词: {question}")
        
        # 位置类问题
        location_match = re.search(r"(\w+)(?:在哪里|在哪个地方|位于何处)", question)
        if location_match:
            entity = location_match.group(1)
            return Predicate("位于", [entity, "X"]), False
        
        # 所有权类问题
        possession_match = re.search(r"(\w+)(?:有什么|拥有什么|持有什么)", question)
        if possession_match:
            entity = possession_match.group(1)
            return Predicate("拥有", [entity, "X"]), False
        
        # 是非类问题
        existence_match = re.search(r"(\w+)(?:在|位于|去了)(\w+)(?:吗)", question)
        if existence_match:
            entity, location = existence_match.groups()
            return Predicate("位于", [entity, location]), False
        
        # 谁在特定位置问题
        who_match = re.search(r"谁(?:在|位于)(\w+)", question)
        if who_match:
            location = who_match.group(1)
            return Predicate("位于", ["X", location]), False
        
        # 属性类问题
        attribute_match = re.search(r"(\w+)(?:是什么|是几|是哪)(\w+)", question)
        if attribute_match:
            entity, attribute_type = attribute_match.groups()
            if "颜色" in attribute_type:
                return Predicate("颜色", [entity, "X"]), False
            elif "大小" in attribute_type:
                return Predicate("大小", [entity, "X"]), False
        
        # 如果无法提取，尝试使用语言模型
        if hasattr(self.language_model, "generate"):
            try:
                # 构造提示词
                prompt = f"""
                请将以下问题转换为查询谓词的形式:
                
                问题: "{question}"
                
                使用以下格式输出(不要添加其他解释):
                谓词: 名称
                参数: 值1, 值2, ...（使用X表示未知变量）
                否定: 是/否
                """
                
                response = self.language_model.generate(prompt, max_new_tokens=50)
                
                # 提取谓词名称
                pred_match = re.search(r"谓词: *(\w+)", response)
                if pred_match:
                    pred_name = pred_match.group(1)
                    
                    # 提取参数
                    args_match = re.search(r"参数: *(.*)", response)
                    args = []
                    if args_match:
                        args_str = args_match.group(1)
                        args = [arg.strip() for arg in args_str.split(",")]
                    
                    # 提取是否为否定
                    neg_match = re.search(r"否定: *(是|否)", response)
                    is_negated = neg_match and neg_match.group(1) == "是"
                    
                    return Predicate(pred_name, args), is_negated
            
            except Exception as e:
                logger.error(f"使用语言模型提取查询谓词时出错: {e}")
        
        # 无法提取
        return None, False


# 简单测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 导入模拟语言模型
    from neusym.models.language_model import MockLanguageModel
    
    # 初始化接口
    lm = MockLanguageModel()
    interface = NeuralSymbolicInterface(lm)
    
    # 测试文本到逻辑形式的转换
    text = "约翰在厨房。玛丽有一本书。"
    facts = interface.text_to_logical_form(text)
    
    print("文本转逻辑形式:")
    for fact in facts:
        print(f"  {fact}")
    
    # 测试逻辑形式到文本的转换
    fact = Fact(Predicate("位于", ["约翰", "厨房"]))
    text = interface.logical_form_to_text(fact)
    print(f"\n逻辑形式转文本: {text}")
    
    # 测试从问题提取查询谓词
    question = "约翰在哪里？"
    predicate, is_negated = interface.extract_query_predicate(question)
    if predicate:
        print(f"\n问题谓词提取: {'¬' if is_negated else ''}{predicate}")
    else:
        print("\n无法从问题中提取谓词") 