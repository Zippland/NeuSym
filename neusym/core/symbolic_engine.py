#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
符号推理引擎模块

提供基础的谓词逻辑处理、事实存储和规则推理能力。
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any, FrozenSet
from collections import defaultdict

logger = logging.getLogger("neusym.symbolic")

class Predicate:
    """谓词类，表示逻辑关系"""
    
    def __init__(self, name: str, args: List[str]):
        """
        初始化谓词
        
        Args:
            name: 谓词名称
            args: 谓词参数列表
        """
        self.name = name
        self.args = args
    
    def __str__(self) -> str:
        args_str = ", ".join(self.args)
        return f"{self.name}({args_str})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Predicate):
            return False
        return self.name == other.name and self.args == other.args
    
    def __hash__(self) -> int:
        return hash((self.name, tuple(self.args)))
    
    def is_variable(self, arg: str) -> bool:
        """检查参数是否为变量"""
        return arg.isupper() or arg == "?" or arg.startswith("?")
    
    def get_variables(self) -> List[str]:
        """获取谓词中的所有变量"""
        return [arg for arg in self.args if self.is_variable(arg)]
    
    def substitute(self, bindings: Dict[str, str]) -> 'Predicate':
        """
        使用绑定替换变量
        
        Args:
            bindings: 变量绑定字典
            
        Returns:
            替换后的谓词
        """
        new_args = []
        for arg in self.args:
            if self.is_variable(arg) and arg in bindings:
                new_args.append(bindings[arg])
            else:
                new_args.append(arg)
        return Predicate(self.name, new_args)


class Fact:
    """事实类，表示一个带有真值的谓词"""
    
    def __init__(self, predicate: Predicate, is_negated: bool = False):
        """
        初始化事实
        
        Args:
            predicate: 谓词
            is_negated: 是否为否定事实
        """
        self.predicate = predicate
        self.is_negated = is_negated
    
    def __str__(self) -> str:
        if self.is_negated:
            return f"¬{self.predicate}"
        return str(self.predicate)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Fact):
            return False
        return (self.predicate == other.predicate and 
                self.is_negated == other.is_negated)
    
    def __hash__(self) -> int:
        return hash((self.predicate, self.is_negated))
    
    def substitute(self, bindings: Dict[str, str]) -> 'Fact':
        """
        使用绑定替换变量
        
        Args:
            bindings: 变量绑定字典
            
        Returns:
            替换后的事实
        """
        new_predicate = self.predicate.substitute(bindings)
        return Fact(new_predicate, self.is_negated)


class Rule:
    """推理规则类，由条件和结论组成"""
    
    def __init__(
        self, 
        conditions: List[Fact], 
        conclusions: List[Fact],
        name: str = ""
    ):
        """
        初始化规则
        
        Args:
            conditions: 条件事实列表
            conclusions: 结论事实列表
            name: 规则名称
        """
        self.conditions = conditions
        self.conclusions = conclusions
        self.name = name
        
        # 提取规则中的所有变量
        self.variables = set()
        for fact in conditions + conclusions:
            self.variables.update(fact.predicate.get_variables())
    
    def __str__(self) -> str:
        conditions_str = " ∧ ".join(str(cond) for cond in self.conditions)
        conclusions_str = " ∧ ".join(str(concl) for concl in self.conclusions)
        rule_str = f"{conditions_str} → {conclusions_str}"
        
        if self.name:
            rule_str = f"{self.name}: {rule_str}"
            
        return rule_str


class SymbolicEngine:
    """符号推理引擎，用于存储事实并进行推理"""
    
    def __init__(self):
        """初始化符号引擎"""
        # 索引保存事实: predicate_name -> args -> [facts]
        self.facts = defaultdict(lambda: defaultdict(list))
        
        # 规则列表
        self.rules = []
        
        # 推理痕迹
        self.reasoning_trace = []
        
        logger.info("符号引擎初始化完成")
    
    def add_fact(self, fact: Fact) -> bool:
        """
        添加事实到知识库
        
        Args:
            fact: 要添加的事实
            
        Returns:
            bool: 是否成功添加（如果已存在则返回False）
        """
        predicate = fact.predicate
        args_tuple = tuple(predicate.args)
        
        # 检查是否已存在
        for existing_fact in self.facts[predicate.name][args_tuple]:
            if existing_fact == fact:
                return False
        
        # 添加事实
        self.facts[predicate.name][args_tuple].append(fact)
        self.reasoning_trace.append(f"添加事实: {fact}")
        
        # 应用规则进行推理
        self._apply_rules()
        
        return True
    
    def add_rule(self, rule: Rule):
        """
        添加规则到引擎
        
        Args:
            rule: 要添加的规则
        """
        self.rules.append(rule)
        self.reasoning_trace.append(f"添加规则: {rule}")
    
    def query(self, predicate: Predicate, negated: bool = False) -> List[Dict[str, str]]:
        """
        查询满足谓词的绑定
        
        Args:
            predicate: 要查询的谓词
            negated: 是否查询否定形式
            
        Returns:
            List[Dict[str, str]]: 变量绑定列表
        """
        self.reasoning_trace.append(f"查询: {'¬' if negated else ''}{predicate}")
        
        variables = predicate.get_variables()
        bindings_list = []
        
        # 如果没有变量，直接检查事实是否存在
        if not variables:
            fact = Fact(predicate, negated)
            pred_name = predicate.name
            args_tuple = tuple(predicate.args)
            
            for existing_fact in self.facts[pred_name][args_tuple]:
                if existing_fact == fact:
                    return [{}]  # 返回空绑定表示匹配成功
            
            return []  # 没有匹配
        
        # 有变量，需要找到所有可能的绑定
        for args_tuple, facts_list in self.facts[predicate.name].items():
            if len(args_tuple) != len(predicate.args):
                continue
                
            # 尝试匹配
            bindings = {}
            match = True
            
            for i, (query_arg, fact_arg) in enumerate(zip(predicate.args, args_tuple)):
                if predicate.is_variable(query_arg):
                    # 变量，需要绑定
                    if query_arg in bindings and bindings[query_arg] != fact_arg:
                        # 变量已绑定且不一致
                        match = False
                        break
                    bindings[query_arg] = fact_arg
                elif query_arg != fact_arg:
                    # 常量不匹配
                    match = False
                    break
            
            # 检查是否有匹配的事实
            if match:
                for fact in facts_list:
                    if fact.is_negated == negated:
                        bindings_list.append(bindings)
                        break
        
        return bindings_list
    
    def _apply_rules(self):
        """应用规则推导新事实"""
        # 简单的前向链接算法
        new_facts_added = True
        
        while new_facts_added:
            new_facts_added = False
            
            for rule in self.rules:
                # 尝试匹配规则条件
                bindings_sets = self._match_conditions(rule.conditions)
                
                for bindings in bindings_sets:
                    # 应用绑定到结论
                    for conclusion in rule.conclusions:
                        new_fact = conclusion.substitute(bindings)
                        
                        # 检查是否存在谓词中的变量未绑定
                        has_unbound_var = False
                        for arg in new_fact.predicate.args:
                            if new_fact.predicate.is_variable(arg):
                                has_unbound_var = True
                                break
                        
                        if not has_unbound_var:
                            # 添加新的事实
                            if self.add_fact(new_fact):
                                new_facts_added = True
                                self.reasoning_trace.append(
                                    f"应用规则 '{rule.name}' 推导: {new_fact}"
                                )
    
    def _match_conditions(self, conditions: List[Fact]) -> List[Dict[str, str]]:
        """
        匹配规则条件
        
        Args:
            conditions: 条件事实列表
            
        Returns:
            List[Dict[str, str]]: 满足条件的变量绑定列表
        """
        if not conditions:
            return [{}]
        
        # 处理第一个条件
        first_condition = conditions[0]
        first_bindings = self.query(
            first_condition.predicate, 
            first_condition.is_negated
        )
        
        if not first_bindings:
            return []
        
        if len(conditions) == 1:
            return first_bindings
        
        # 递归处理剩余条件
        result_bindings = []
        remaining_conditions = conditions[1:]
        
        for bindings in first_bindings:
            # 替换剩余条件中的变量
            substituted_conditions = [
                cond.substitute(bindings) for cond in remaining_conditions
            ]
            
            # 递归匹配
            sub_bindings_list = self._match_conditions(substituted_conditions)
            
            # 合并绑定
            for sub_bindings in sub_bindings_list:
                merged_bindings = bindings.copy()
                for var, val in sub_bindings.items():
                    merged_bindings[var] = val
                result_bindings.append(merged_bindings)
        
        return result_bindings
    
    def get_reasoning_trace(self) -> List[str]:
        """获取推理过程的跟踪记录"""
        return self.reasoning_trace
    
    def reset(self):
        """重置引擎状态"""
        self.facts = defaultdict(lambda: defaultdict(list))
        self.reasoning_trace = []
        logger.info("符号引擎状态已重置")


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化符号引擎
    engine = SymbolicEngine()
    
    # 添加一些规则
    location_rule = Rule(
        conditions=[
            Fact(Predicate("位于", ["X", "Y"])),
            Fact(Predicate("位于", ["Y", "Z"]))
        ],
        conclusions=[
            Fact(Predicate("位于", ["X", "Z"]))
        ],
        name="位置传递规则"
    )
    engine.add_rule(location_rule)
    
    # 添加一些事实
    engine.add_fact(Fact(Predicate("位于", ["约翰", "厨房"])))
    engine.add_fact(Fact(Predicate("位于", ["厨房", "房子"])))
    
    # 执行查询
    results = engine.query(Predicate("位于", ["约翰", "Z"]))
    
    print(f"查询结果:")
    for binding in results:
        if "Z" in binding:
            print(f"  约翰位于: {binding['Z']}")
    
    # 打印推理跟踪
    print("\n推理过程:")
    for step in engine.get_reasoning_trace():
        print(f"  {step}") 