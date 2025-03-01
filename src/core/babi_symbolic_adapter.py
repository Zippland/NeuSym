#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bAbI任务符号推理适配器

为bAbI任务提供特定的符号推理规则和模式识别。
优化符号引擎对不同类型bAbI任务的处理能力。
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any

from .symbolic_engine import SymbolicEngine, Predicate, Fact, Rule

logger = logging.getLogger("neusym.babi_adapter")

class BabiSymbolicAdapter:
    """bAbI任务的符号推理适配器"""
    
    def __init__(self, symbolic_engine: SymbolicEngine):
        """
        初始化bAbI符号适配器
        
        Args:
            symbolic_engine: 符号引擎实例
        """
        self.engine = symbolic_engine
        self.task_specific_rules = {}
        logger.info("bAbI符号适配器初始化完成")
        
    def configure_for_task(self, task_id: int):
        """
        为特定的bAbI任务配置符号引擎
        
        Args:
            task_id: bAbI任务ID (1-20)
        """
        logger.info(f"为bAbI任务{task_id}配置符号引擎")
        
        # 重置引擎
        self.engine.reset()
        
        # 根据任务类型添加特定规则
        if task_id == 1:
            self._configure_task1()
        elif task_id == 2:
            self._configure_task2()
        elif task_id == 3:
            self._configure_task3()
        elif task_id in (4, 5):
            self._configure_task4_5()
        elif task_id == 6:
            self._configure_task6()
        elif task_id == 7:
            self._configure_task7()
        elif task_id == 8:
            self._configure_task8()
        elif task_id in (9, 10):
            self._configure_task9_10()
        elif task_id == 11:
            self._configure_task11()
        elif task_id == 12:
            self._configure_task12()
        elif task_id == 13:
            self._configure_task13()
        elif task_id == 14:
            self._configure_task14()
        elif task_id == 15:
            self._configure_task15()
        elif task_id == 16:
            self._configure_task16()
        elif task_id == 17:
            self._configure_task17()
        elif task_id == 18:
            self._configure_task18()
        elif task_id == 19:
            self._configure_task19()
        elif task_id == 20:
            self._configure_task20()
        else:
            logger.warning(f"未找到任务{task_id}的特定配置，使用通用配置")
            self._configure_generic()
            
        logger.info(f"任务{task_id}配置完成，共添加{len(self.task_specific_rules)}条特定规则")

    def _configure_generic(self):
        """配置通用规则，适用于所有任务类型"""
        # 添加基本的位置关系推理规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("位于", ["X", "Y"]))
                ],
                conclusions=[
                    Fact(Predicate("位于", ["X", "Y"]))
                ],
                name="位置传递规则"
            )
        )
        
        # 添加基本的所有权关系规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("拥有", ["X", "Y"]))
                ],
                conclusions=[
                    Fact(Predicate("拥有", ["X", "Y"]))
                ],
                name="所有权规则"
            )
        )
        
        self.task_specific_rules["generic"] = ["位置传递规则", "所有权规则"]
    
    def _configure_task1(self):
        """配置任务1：单一支持事实"""
        # 任务1只需要识别单一事实，无需特殊规则
        self._configure_generic()
        
    def _configure_task2(self):
        """配置任务2：两个支持事实"""
        # 任务2需要多跳推理，添加位置传递规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("位于", ["X", "Y"])),
                    Fact(Predicate("位于", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("位于", ["X", "Z"]))
                ],
                name="位置传递性规则"
            )
        )
        
        self.task_specific_rules["task2"] = ["位置传递性规则"]
    
    def _configure_task3(self):
        """配置任务3：三个支持事实"""
        # 配置两个事实的规则
        self._configure_task2()
        
        # 添加三跳推理支持
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("位于", ["X", "Y"])),
                    Fact(Predicate("位于", ["Y", "Z"])),
                    Fact(Predicate("位于", ["Z", "W"]))
                ],
                conclusions=[
                    Fact(Predicate("位于", ["X", "W"]))
                ],
                name="三跳位置传递规则"
            )
        )
        
        self.task_specific_rules["task3"] = ["位置传递性规则", "三跳位置传递规则"]
    
    def _configure_task4_5(self):
        """配置任务4-5：两个和三个论点关系"""
        # 添加论点组合关系
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("左侧", ["X", "Y"])),
                ],
                conclusions=[
                    Fact(Predicate("右侧", ["Y", "X"]))
                ],
                name="左右互逆规则"
            )
        )
        
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("右侧", ["X", "Y"])),
                ],
                conclusions=[
                    Fact(Predicate("左侧", ["Y", "X"]))
                ],
                name="右左互逆规则"
            )
        )
        
        # 传递规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("左侧", ["X", "Y"])),
                    Fact(Predicate("左侧", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("左侧", ["X", "Z"]))
                ],
                name="左侧传递规则"
            )
        )
        
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("右侧", ["X", "Y"])),
                    Fact(Predicate("右侧", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("右侧", ["X", "Z"]))
                ],
                name="右侧传递规则"
            )
        )
        
        self.task_specific_rules["task4_5"] = ["左右互逆规则", "右左互逆规则", "左侧传递规则", "右侧传递规则"]
    
    def _configure_task6(self):
        """配置任务6：是非问题"""
        # 添加否定逻辑处理
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("在", ["X", "Y"]), is_negated=True)
                ],
                conclusions=[
                    Fact(Predicate("不在", ["X", "Y"]))
                ],
                name="否定转换规则"
            )
        )
        
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("不在", ["X", "Y"]))
                ],
                conclusions=[
                    Fact(Predicate("在", ["X", "Y"]), is_negated=True)
                ],
                name="否定转换规则2"
            )
        )
        
        self.task_specific_rules["task6"] = ["否定转换规则", "否定转换规则2"]
    
    def _configure_task7(self):
        """配置任务7：计数"""
        # 计数任务不需要特定的符号规则，依赖自然语言处理部分
        self._configure_generic()
    
    def _configure_task8(self):
        """配置任务8：列表/集合"""
        # 列表任务不需要特定的符号规则，依赖自然语言处理部分
        self._configure_generic()
    
    def _configure_task9_10(self):
        """配置任务9-10：简单否定和明确否定"""
        # 添加否定逻辑
        self._configure_task6()
        
        # 添加补充规则处理明确否定
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("是", ["X", "Y"]), is_negated=True)
                ],
                conclusions=[
                    Fact(Predicate("不是", ["X", "Y"]))
                ],
                name="属性否定规则"
            )
        )
        
        self.task_specific_rules["task9_10"] = ["否定转换规则", "否定转换规则2", "属性否定规则"]
    
    def _configure_task11(self):
        """配置任务11：基本归纳"""
        # 归纳任务主要通过语言模型处理
        self._configure_generic()
    
    def _configure_task12(self):
        """配置任务12：推理"""
        # 添加推理规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("是", ["X", "Y"])),
                    Fact(Predicate("需要", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("需要", ["X", "Z"]))
                ],
                name="需求推理规则"
            )
        )
        
        self.task_specific_rules["task12"] = ["需求推理规则"]
    
    def _configure_task13(self):
        """配置任务13：复合事实"""
        # 添加复合事实处理规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("颜色", ["X", "Y"])),
                    Fact(Predicate("形状", ["X", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("描述", ["X", "Y与Z"]))
                ],
                name="属性组合规则"
            )
        )
        
        self.task_specific_rules["task13"] = ["属性组合规则"]
    
    def _configure_task14(self):
        """配置任务14：时间推理"""
        # 添加时间先后关系规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("之后", ["X", "Y"])),
                ],
                conclusions=[
                    Fact(Predicate("之前", ["Y", "X"]))
                ],
                name="时间互逆规则"
            )
        )
        
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("之前", ["X", "Y"])),
                    Fact(Predicate("之前", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("之前", ["X", "Z"]))
                ],
                name="时间传递规则"
            )
        )
        
        self.task_specific_rules["task14"] = ["时间互逆规则", "时间传递规则"]
    
    def _configure_task15(self):
        """配置任务15：基本演绎"""
        # 添加演绎推理规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("是", ["X", "Y"])),
                    Fact(Predicate("都是", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("是", ["X", "Z"]))
                ],
                name="基本演绎规则"
            )
        )
        
        self.task_specific_rules["task15"] = ["基本演绎规则"]
    
    def _configure_task16(self):
        """配置任务16：基本因果"""
        # 添加因果关系规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("导致", ["X", "Y"])),
                    Fact(Predicate("发生", ["X"]))
                ],
                conclusions=[
                    Fact(Predicate("发生", ["Y"]))
                ],
                name="因果关系规则"
            )
        )
        
        self.task_specific_rules["task16"] = ["因果关系规则"]
    
    def _configure_task17(self):
        """配置任务17：时态关系"""
        # 添加时态规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("现在在", ["X", "Y"]))
                ],
                conclusions=[
                    Fact(Predicate("在", ["X", "Y"]))
                ],
                name="现在时规则"
            )
        )
        
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("曾经在", ["X", "Y"])),
                    Fact(Predicate("现在不在", ["X", "Y"]))
                ],
                conclusions=[
                    Fact(Predicate("在", ["X", "Y"]), is_negated=True)
                ],
                name="过去时规则"
            )
        )
        
        self.task_specific_rules["task17"] = ["现在时规则", "过去时规则"]
    
    def _configure_task18(self):
        """配置任务18：大小比较"""
        # 添加大小比较规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("大于", ["X", "Y"])),
                ],
                conclusions=[
                    Fact(Predicate("小于", ["Y", "X"]))
                ],
                name="大小互逆规则"
            )
        )
        
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("大于", ["X", "Y"])),
                    Fact(Predicate("大于", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("大于", ["X", "Z"]))
                ],
                name="大小传递规则"
            )
        )
        
        self.task_specific_rules["task18"] = ["大小互逆规则", "大小传递规则"]
    
    def _configure_task19(self):
        """配置任务19：路径寻找"""
        # 添加路径规则
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("连接", ["X", "Y"])),
                ],
                conclusions=[
                    Fact(Predicate("连接", ["Y", "X"]))
                ],
                name="路径双向规则"
            )
        )
        
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("连接", ["X", "Y"])),
                    Fact(Predicate("连接", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("可达", ["X", "Z"]))
                ],
                name="路径传递规则"
            )
        )
        
        self.engine.add_rule(
            Rule(
                conditions=[
                    Fact(Predicate("可达", ["X", "Y"])),
                    Fact(Predicate("连接", ["Y", "Z"]))
                ],
                conclusions=[
                    Fact(Predicate("可达", ["X", "Z"]))
                ],
                name="路径扩展规则"
            )
        )
        
        self.task_specific_rules["task19"] = ["路径双向规则", "路径传递规则", "路径扩展规则"]
    
    def _configure_task20(self):
        """配置任务20：代理动机"""
        # 代理动机任务需要复杂的推理，主要依靠神经网络部分
        self._configure_generic()

    def extract_predicates_from_text(self, text: str) -> List[Fact]:
        """
        从自然语言文本中提取谓词逻辑事实
        
        Args:
            text: 输入文本
            
        Returns:
            提取的事实列表
        """
        facts = []
        
        # 位置关系模式
        location_pattern = r"(\w+)(?:在|位于|去了)(\w+)"
        for match in re.finditer(location_pattern, text):
            entity, location = match.groups()
            pred = Predicate("位于", [entity, location])
            facts.append(Fact(pred))
        
        # 拥有关系模式
        possession_pattern = r"(\w+)(?:拥有|有|持有)(\w+)"
        for match in re.finditer(possession_pattern, text):
            entity, item = match.groups()
            pred = Predicate("拥有", [entity, item])
            facts.append(Fact(pred))
        
        # 给予关系模式
        gave_pattern = r"(\w+)(?:给了|给|赠送)(\w+)(?:给|到)(\w+)"
        for match in re.finditer(gave_pattern, text):
            giver, item, receiver = match.groups()
            pred1 = Predicate("给予", [giver, receiver, item])
            pred2 = Predicate("拥有", [receiver, item])
            facts.extend([Fact(pred1), Fact(pred2)])
        
        # 空间关系模式
        spatial_pattern = r"(\w+)(?:在|位于)(\w+)(?:的)(\w+边)"
        for match in re.finditer(spatial_pattern, text):
            entity, reference, direction = match.groups()
            pred = Predicate(direction, [entity, reference])
            facts.append(Fact(pred))
        
        # 否定模式
        negative_pattern = r"(\w+)(?:不在|不位于|没去)(\w+)"
        for match in re.finditer(negative_pattern, text):
            entity, location = match.groups()
            pred = Predicate("位于", [entity, location])
            facts.append(Fact(pred, is_negated=True))
        
        # 如果没有提取到事实，尝试直接将整个句子作为事实
        if not facts:
            pred = Predicate("陈述", [text])
            facts.append(Fact(pred))
        
        return facts
    
    def map_question_to_query(self, question: str) -> Dict:
        """
        将自然语言问题映射为符号查询
        
        Args:
            question: 问题文本
            
        Returns:
            查询信息
        """
        query_info = {"type": "unknown", "parameters": {}}
        
        # 位置问题
        location_pattern = r"(\w+)(?:在哪里|在哪个地方|位于何处)"
        match = re.search(location_pattern, question)
        if match:
            entity = match.group(1)
            query_info = {
                "type": "location",
                "parameters": {"entity": entity},
                "predicate": Predicate("位于", [entity, "?"]),
            }
            return query_info
        
        # 拥有关系问题
        possession_pattern = r"(\w+)(?:有什么|拥有什么|持有什么)"
        match = re.search(possession_pattern, question)
        if match:
            entity = match.group(1)
            query_info = {
                "type": "possession",
                "parameters": {"entity": entity},
                "predicate": Predicate("拥有", [entity, "?"]),
            }
            return query_info
        
        # 存在性问题 (是/否)
        existence_pattern = r"(\w+)(?:在|位于|去了)(\w+)(?:吗)"
        match = re.search(existence_pattern, question)
        if match:
            entity, location = match.groups()
            query_info = {
                "type": "existence",
                "parameters": {"entity": entity, "location": location},
                "predicate": Predicate("位于", [entity, location]),
            }
            return query_info
        
        # 未识别具体模式，返回通用查询
        query_info = {
            "type": "general",
            "parameters": {"question": question},
            "predicate": Predicate("回答", [question, "?"]),
        }
        
        return query_info
    
    def process_question(self, question: str) -> str:
        """
        处理问题并返回答案
        
        Args:
            question: 问题文本
            
        Returns:
            答案文本
        """
        # 解析问题为查询
        query_info = self.map_question_to_query(question)
        
        if query_info["type"] == "location":
            # 处理位置问题
            entity = query_info["parameters"]["entity"]
            results = self.engine.query(Predicate("位于", [entity, "X"]))
            
            if results:
                # 返回满足条件的位置
                locations = [binding["X"] for binding in results]
                if len(locations) == 1:
                    return locations[0]
                else:
                    return "、".join(locations)
            else:
                return "未知"
                
        elif query_info["type"] == "possession":
            # 处理拥有关系问题
            entity = query_info["parameters"]["entity"]
            results = self.engine.query(Predicate("拥有", [entity, "X"]))
            
            if results:
                # 返回满足条件的物品
                items = [binding["X"] for binding in results]
                if len(items) == 1:
                    return items[0]
                else:
                    return "、".join(items)
            else:
                return "没有物品"
                
        elif query_info["type"] == "existence":
            # 处理存在性问题
            entity = query_info["parameters"]["entity"]
            location = query_info["parameters"]["location"]
            results = self.engine.query(Predicate("位于", [entity, location]))
            
            if results:
                return "是"
            else:
                # 检查是否有明确的否定
                neg_results = self.engine.query(Predicate("位于", [entity, location]), negated=True)
                if neg_results:
                    return "否"
                else:
                    return "未知"
        
        # 对于一般性问题，目前只能返回未知
        return "未能找到答案"


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化符号引擎
    engine = SymbolicEngine()
    
    # 初始化bAbI适配器
    adapter = BabiSymbolicAdapter(engine)
    
    # 配置任务1
    adapter.configure_for_task(1)
    
    # 添加一些事实
    facts = adapter.extract_predicates_from_text("约翰在厨房。")
    for fact in facts:
        engine.add_fact(fact)
    
    # 测试问题
    answer = adapter.process_question("约翰在哪里？")
    print(f"问题: 约翰在哪里？\n答案: {answer}") 