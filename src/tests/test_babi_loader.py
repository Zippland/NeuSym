#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试bAbI数据加载器
"""

import os
import sys
import logging
import unittest
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.babi_loader import BabiDataLoader

class TestBabiLoader(unittest.TestCase):
    """测试bAbI数据加载器"""
    
    def setUp(self):
        """测试前设置"""
        logging.basicConfig(level=logging.INFO)
        self.loader = BabiDataLoader()
    
    def test_task_names(self):
        """测试任务名称映射"""
        for task_id in range(1, 21):
            name = self.loader.get_task_name(task_id)
            self.assertIsNotNone(name)
            self.assertNotEqual(name, f"未知任务{task_id}")
    
    def test_get_all_tasks(self):
        """测试获取所有任务ID"""
        tasks = self.loader.get_all_tasks()
        self.assertEqual(len(tasks), 20)
        self.assertEqual(tasks, list(range(1, 21)))
    
    def test_data_download(self):
        """测试数据下载功能"""
        try:
            result = self.loader.download_dataset()
            self.assertTrue(result)
            self.assertTrue(self.loader.downloaded)
            
            # 检查数据目录是否存在
            tasks_dir = self.loader.data_dir / "tasks_1-20_v1-2"
            self.assertTrue(tasks_dir.exists())
            
        except Exception as e:
            self.fail(f"下载数据时出错: {e}")
    
    def test_load_task(self):
        """测试加载任务数据"""
        # 仅当数据已下载时测试
        if not self.loader.downloaded:
            try:
                self.loader.download_dataset()
            except:
                self.skipTest("无法下载数据集，跳过此测试")
        
        # 测试加载任务1
        task_data = self.loader.load_task(1, "train")
        self.assertIsNotNone(task_data)
        self.assertGreater(len(task_data), 0)
        
        # 检查数据结构
        first_story = task_data[0]
        self.assertIn("statements", first_story)
        self.assertIn("questions", first_story)
        
        # 检查语句结构
        if first_story["statements"]:
            first_stmt = first_story["statements"][0]
            self.assertIn("id", first_stmt)
            self.assertIn("text", first_stmt)
        
        # 检查问题结构
        if first_story["questions"]:
            first_q = first_story["questions"][0]
            self.assertIn("id", first_q)
            self.assertIn("text", first_q)
            self.assertIn("answer", first_q)
            self.assertIn("supporting", first_q)
    
    def test_prepare_for_symbolic_engine(self):
        """测试准备符号引擎数据"""
        # 仅当数据已下载时测试
        if not self.loader.downloaded:
            try:
                self.loader.download_dataset()
            except:
                self.skipTest("无法下载数据集，跳过此测试")
        
        # 加载数据
        task_data = self.loader.load_task(1, "train")
        if not task_data:
            self.skipTest("无法加载任务数据")
        
        # 测试转换
        first_story = task_data[0]
        facts, queries = self.loader.prepare_for_symbolic_engine(first_story)
        
        # 检查结果
        self.assertIsNotNone(facts)
        self.assertIsNotNone(queries)
        self.assertEqual(len(facts), len(first_story["statements"]))
        self.assertEqual(len(queries), len(first_story["questions"]))
        
        # 检查查询结构
        if queries:
            first_query = queries[0]
            self.assertIn("question", first_query)
            self.assertIn("answer", first_query)
            self.assertIn("supporting_facts", first_query)

if __name__ == "__main__":
    unittest.main() 