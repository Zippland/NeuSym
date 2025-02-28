#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合推理控制器模块

协调神经网络和符号系统的交互，智能决策使用哪种推理方式。
根据查询和知识库的状态，动态选择最适合的推理路径。
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any

from .symbolic_engine import SymbolicEngine, Fact, Predicate
from .neural_symbolic_interface import NeuralSymbolicInterface

logger = logging.getLogger("neusym.controller")

class HybridReasoningController:
    """混合推理控制器，协调神经网络和符号系统"""
    
    def __init__(
        self,
        language_model,
        symbolic_engine: SymbolicEngine,
        neural_symbolic_interface: NeuralSymbolicInterface,
        uncertainty_threshold: float = 0.6
    ):
        """
        初始化混合推理控制器
        
        Args:
            language_model: 语言模型实例
            symbolic_engine: 符号引擎实例
            neural_symbolic_interface: 神经-符号接口实例
            uncertainty_threshold: 不确定性阈值，高于此值时优先使用符号推理
        """
        self.language_model = language_model
        self.symbolic_engine = symbolic_engine
        self.interface = neural_symbolic_interface
        self.uncertainty_threshold = uncertainty_threshold
        
        # 推理跟踪
        self.reasoning_trace = []
        
        logger.info("混合推理控制器初始化完成")
    
    def process_query(self, query_text: str) -> str:
        """
        处理查询
        
        Args:
            query_text: 查询文本
            
        Returns:
            回答文本
        """
        self.reasoning_trace = []
        self.reasoning_trace.append(f"接收查询: {query_text}")
        
        # 1. 尝试提取查询谓词
        predicate, is_negated = self.interface.extract_query_predicate(query_text)
        
        # 2. 如果成功提取查询谓词，尝试使用符号推理
        if predicate:
            self.reasoning_trace.append(f"提取查询谓词: {predicate}, 否定={is_negated}")
            
            # 执行符号查询
            bindings = self.symbolic_engine.query(predicate, is_negated)
            
            if bindings:
                self.reasoning_trace.append(f"符号推理成功: 找到{len(bindings)}个结果")
                
                # 处理查询结果
                answer = self._process_symbolic_results(predicate, bindings, query_text)
                self.reasoning_trace.append(f"符号推理回答: {answer}")
                return answer
            else:
                self.reasoning_trace.append("符号推理未找到结果，尝试神经推理")
        else:
            self.reasoning_trace.append("无法提取查询谓词，使用神经推理")
        
        # 3. 使用神经推理作为后备方案
        answer = self._neural_reasoning(query_text)
        self.reasoning_trace.append(f"神经推理回答: {answer}")
        return answer
    
    def _process_symbolic_results(
        self, 
        query_predicate: Predicate, 
        bindings: List[Dict[str, str]],
        original_query: str
    ) -> str:
        """
        处理符号查询结果
        
        Args:
            query_predicate: 查询谓词
            bindings: 变量绑定列表
            original_query: 原始查询文本
            
        Returns:
            格式化的回答
        """
        # 获取查询变量
        variables = query_predicate.get_variables()
        
        # 如果没有变量（是/否查询）
        if not variables:
            return "是" if bindings else "否"
        
        # 如果是位置查询
        if query_predicate.name == "位于" and "X" in query_predicate.args:
            if query_predicate.args[0] == "X":  # 谁在某地？
                location = query_predicate.args[1]
                entities = [binding["X"] for binding in bindings if "X" in binding]
                
                if entities:
                    if len(entities) == 1:
                        return entities[0]
                    else:
                        return "、".join(entities)
                else:
                    return "没有人"
            else:  # 某人在哪里？
                entity = query_predicate.args[0]
                locations = [binding["X"] for binding in bindings if "X" in binding]
                
                if locations:
                    if len(locations) == 1:
                        return locations[0]
                    else:
                        return "、".join(locations)
                else:
                    return "未知地点"
        
        # 如果是拥有关系查询
        if query_predicate.name == "拥有" and "X" in query_predicate.args:
            if query_predicate.args[0] == "X":  # 谁拥有某物？
                item = query_predicate.args[1]
                owners = [binding["X"] for binding in bindings if "X" in binding]
                
                if owners:
                    if len(owners) == 1:
                        return owners[0]
                    else:
                        return "、".join(owners)
                else:
                    return "没有人"
            else:  # 某人拥有什么？
                owner = query_predicate.args[0]
                items = [binding["X"] for binding in bindings if "X" in binding]
                
                if items:
                    if len(items) == 1:
                        return items[0]
                    else:
                        return "、".join(items)
                else:
                    return "没有物品"
        
        # 通用变量结果处理
        var_name = variables[0]
        values = []
        
        for binding in bindings:
            if var_name in binding:
                values.append(binding[var_name])
        
        if values:
            if len(values) == 1:
                return values[0]
            else:
                return "、".join(values)
        
        return "未找到答案"
    
    def _neural_reasoning(self, query_text: str) -> str:
        """
        使用神经推理处理查询
        
        Args:
            query_text: 查询文本
            
        Returns:
            推理结果
        """
        self.reasoning_trace.append("执行神经推理")
        
        # 收集知识库中的相关事实
        facts = []
        for pred_name, args_dict in self.symbolic_engine.facts.items():
            for args_tuple, facts_list in args_dict.items():
                facts.extend(facts_list)
        
        # 如果知识库中有事实，尝试结合事实生成答案
        if facts:
            self.reasoning_trace.append(f"使用{len(facts)}个事实进行神经推理")
            
            # 将事实转换为文本
            fact_texts = []
            for fact in facts:
                fact_text = self.interface.logical_form_to_text(fact)
                fact_texts.append(fact_text)
            
            # 结合知识生成回答
            context = ". ".join(fact_texts)
            
            try:
                prompt = f"""
                根据以下信息回答问题:
                
                已知信息:
                {context}
                
                问题: {query_text}
                
                回答:
                """
                
                answer = self.language_model.generate(prompt, max_new_tokens=100)
                return answer.strip()
            except Exception as e:
                logger.error(f"神经推理生成回答时出错: {e}")
                return "我无法回答这个问题"
        else:
            # 没有相关事实，直接使用语言模型回答
            self.reasoning_trace.append("没有相关事实，使用语言模型直接回答")
            
            try:
                prompt = f"问题: {query_text}\n\n回答:"
                answer = self.language_model.generate(prompt, max_new_tokens=100)
                return answer.strip()
            except Exception as e:
                logger.error(f"语言模型直接回答时出错: {e}")
                return "我无法回答这个问题"
    
    def add_fact(self, fact_text: str) -> bool:
        """
        添加事实到知识库
        
        Args:
            fact_text: 事实文本
            
        Returns:
            bool: 是否成功添加
        """
        # 解析事实
        facts = self.interface.text_to_logical_form(fact_text)
        
        if not facts:
            logger.warning(f"无法从文本中提取事实: {fact_text}")
            return False
        
        # 添加事实到符号引擎
        success = False
        for fact in facts:
            if self.symbolic_engine.add_fact(fact):
                success = True
                logger.info(f"成功添加事实: {fact}")
            else:
                logger.info(f"事实已存在: {fact}")
        
        return success
    
    def add_facts_batch(self, texts: List[str]) -> int:
        """
        批量添加事实到知识库
        
        Args:
            texts: 事实文本列表
            
        Returns:
            int: 成功添加的事实数量
        """
        count = 0
        for text in texts:
            if self.add_fact(text):
                count += 1
        
        return count
    
    def get_reasoning_trace(self) -> List[str]:
        """获取推理过程的跟踪记录"""
        return self.reasoning_trace
    
    def reset(self):
        """重置控制器状态"""
        self.symbolic_engine.reset()
        self.reasoning_trace = []
        logger.info("混合推理控制器状态已重置")


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 导入必要的组件
    from neusym.models.language_model import MockLanguageModel
    
    # 初始化组件
    lm = MockLanguageModel()
    symbolic_engine = SymbolicEngine()
    interface = NeuralSymbolicInterface(lm)
    
    # 初始化控制器
    controller = HybridReasoningController(
        language_model=lm,
        symbolic_engine=symbolic_engine,
        neural_symbolic_interface=interface
    )
    
    # 添加一些事实
    controller.add_fact("约翰在厨房")
    controller.add_fact("玛丽在客厅")
    controller.add_fact("厨房在房子里")
    
    # 处理查询
    query = "约翰在哪里？"
    answer = controller.process_query(query)
    
    print(f"问题: {query}")
    print(f"回答: {answer}")
    
    # 打印推理过程
    print("\n推理过程:")
    for step in controller.get_reasoning_trace():
        print(f"- {step}") 