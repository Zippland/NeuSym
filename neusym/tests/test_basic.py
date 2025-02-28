#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基本功能测试脚本
"""

import unittest
import logging

from neusym.core.symbolic_engine import Predicate, Fact, Rule, SymbolicEngine

# 禁用日志输出
logging.disable(logging.CRITICAL)

class TestSymbolicEngine(unittest.TestCase):
    """测试符号推理引擎的基本功能"""
    
    def setUp(self):
        """测试前准备"""
        self.engine = SymbolicEngine()
        
        # 添加一些基本事实
        p1 = Predicate("人", ["苏格拉底"])
        f1 = Fact(p1)
        self.engine.add_fact(f1)
        
        p2 = Predicate("人", ["柏拉图"])
        f2 = Fact(p2)
        self.engine.add_fact(f2)
        
        # 添加规则：所有人都是凡人
        condition = Fact(Predicate("人", ["?x"]))
        conclusion = Fact(Predicate("凡人", ["?x"]))
        r1 = Rule([condition], [conclusion], "人都是凡人")
        self.engine.add_rule(r1)
    
    def test_fact_addition(self):
        """测试添加事实"""
        # 添加新事实
        p3 = Predicate("希腊人", ["苏格拉底"])
        f3 = Fact(p3)
        self.engine.add_fact(f3)
        
        # 验证事实数量
        self.assertEqual(len(self.engine.kb.facts), 3)
    
    def test_rule_addition(self):
        """测试添加规则"""
        # 添加新规则
        condition = Fact(Predicate("希腊人", ["?x"]))
        conclusion = Fact(Predicate("会说希腊语", ["?x"]))
        r2 = Rule([condition], [conclusion], "希腊人会说希腊语")
        self.engine.add_rule(r2)
        
        # 验证规则数量
        self.assertEqual(len(self.engine.kb.rules), 2)
    
    def test_forward_chaining(self):
        """测试前向链接推理"""
        # 执行前向链接推理
        new_facts = self.engine.forward_chaining()
        
        # 验证是否生成了新事实
        self.assertTrue(len(new_facts) > 0)
    
    def test_query(self):
        """测试查询功能"""
        # 查询苏格拉底是否是凡人
        query = Fact(Predicate("凡人", ["苏格拉底"]))
        result, confidence, _ = self.engine.query(query)
        
        # 验证查询结果
        self.assertTrue(result)
        self.assertGreater(confidence, 0)
    
    def test_unification(self):
        """测试谓词统一"""
        # 创建模式和目标谓词
        pattern = Predicate("人", ["?x"])
        target = Predicate("人", ["苏格拉底"])
        
        # 执行统一
        bindings = self.engine.unify(pattern, target)
        
        # 验证绑定结果
        self.assertIsNotNone(bindings)
        self.assertEqual(bindings["?x"], "苏格拉底")
    
    def test_apply_bindings(self):
        """测试应用变量绑定"""
        # 创建谓词和绑定
        predicate = Predicate("拥有", ["?x", "?y"])
        bindings = {"?x": "苏格拉底", "?y": "智慧"}
        
        # 应用绑定
        result = self.engine.apply_bindings(predicate, bindings)
        
        # 验证结果
        self.assertEqual(result.name, "拥有")
        self.assertEqual(result.arguments, ["苏格拉底", "智慧"])
    
    def test_clear_knowledge_base(self):
        """测试清空知识库"""
        # 清空知识库
        self.engine.clear_knowledge_base()
        
        # 验证知识库是否为空
        self.assertEqual(len(self.engine.kb.facts), 0)
        self.assertEqual(len(self.engine.kb.rules), 0)

if __name__ == "__main__":
    unittest.main() 