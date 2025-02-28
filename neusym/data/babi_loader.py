#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bAbI数据集加载器

处理Facebook AI Research的bAbI任务数据集，提供数据加载、预处理和格式化功能。
bAbI数据集包含20个任务，每个任务测试特定类型的推理能力。
"""

import os
import re
import logging
import urllib.request
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union, Any

import numpy as np

logger = logging.getLogger("neusym.data")

# bAbI数据集URL
BABI_DATA_URL = "https://dl.fbaipublicfiles.com/babi/tasks_1-20_v1-2.tar.gz"

# 任务名称映射
BABI_TASKS = {
    1: "单一支持事实",
    2: "两个支持事实",
    3: "三个支持事实",
    4: "两个论点关系",
    5: "三个论点关系",
    6: "是非问题",
    7: "计数",
    8: "列表/集合",
    9: "简单否定",
    10: "明确否定",
    11: "基本归纳",
    12: "推理",
    13: "复合事实",
    14: "时间推理",
    15: "基本演绎",
    16: "基本因果",
    17: "时态关系",
    18: "大小比较",
    19: "路径寻找",
    20: "代理动机"
}

class BabiDataLoader:
    """bAbI数据集加载和处理类"""
    
    def __init__(self, data_dir: Optional[str] = None, language: str = "en"):
        """
        初始化bAbI数据加载器
        
        Args:
            data_dir: 数据存储目录，如果为None，则使用默认目录
            language: 数据集语言，"en"或"cn"
        """
        if data_dir is None:
            # 使用默认目录：项目根目录下的data/babi
            self.data_dir = Path(__file__).resolve().parent / "babi"
        else:
            self.data_dir = Path(data_dir)
        
        self.language = language
        self.downloaded = False
        
        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"bAbI数据加载器初始化完成，数据目录: {self.data_dir}")
    
    def download_dataset(self, force: bool = False) -> bool:
        """
        下载bAbI数据集
        
        Args:
            force: 是否强制重新下载
            
        Returns:
            bool: 下载是否成功
        """
        # 检查数据是否已经下载
        tasks_dir = self.data_dir / "tasks_1-20_v1-2"
        if tasks_dir.exists() and not force:
            logger.info("bAbI数据集已下载")
            self.downloaded = True
            return True
        
        logger.info(f"开始下载bAbI数据集: {BABI_DATA_URL}")
        
        # 下载tar.gz文件
        tar_path = self.data_dir / "babi_tasks.tar.gz"
        try:
            urllib.request.urlretrieve(BABI_DATA_URL, tar_path)
            
            # 解压文件
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)
            
            # 删除tar文件
            tar_path.unlink()
            
            self.downloaded = True
            logger.info("bAbI数据集下载并解压完成")
            return True
            
        except Exception as e:
            logger.error(f"下载bAbI数据集时出错: {e}")
            return False
    
    def get_task_path(self, task_id: int) -> Path:
        """
        获取特定任务数据的路径
        
        Args:
            task_id: 任务ID (1-20)
            
        Returns:
            Path: 任务数据路径
        """
        if not 1 <= task_id <= 20:
            raise ValueError(f"任务ID必须在1-20之间，收到: {task_id}")
        
        base_path = self.data_dir / "tasks_1-20_v1-2" / "en"
        if self.language == "cn":
            base_path = self.data_dir / "tasks_1-20_v1-2" / "cn"
        
        task_path = base_path / f"qa{task_id}_"
        return task_path
    
    def load_task(self, task_id: int, split: str = "train") -> List[Dict]:
        """
        加载特定任务的数据
        
        Args:
            task_id: 任务ID (1-20)
            split: 数据集划分，"train"或"test"
            
        Returns:
            List[Dict]: 加载的数据，每个元素包含一个故事和问题
        """
        if not self.downloaded:
            success = self.download_dataset()
            if not success:
                raise RuntimeError("无法加载数据集，请检查下载错误")
        
        task_path = self.get_task_path(task_id)
        file_path = task_path / f"{split}.txt"
        
        if not file_path.exists():
            file_path = task_path / f"{split}_{self.language}.txt"
            if not file_path.exists():
                file_path = task_path / f"{split}_en.txt"
                if not file_path.exists():
                    raise FileNotFoundError(f"找不到任务{task_id}的{split}数据文件")
        
        logger.info(f"加载任务{task_id}的{split}数据: {file_path}")
        
        # 解析数据文件
        return self._parse_babi_file(file_path)
    
    def _parse_babi_file(self, file_path: Path) -> List[Dict]:
        """
        解析bAbI格式的文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Dict]: 解析后的数据
        """
        stories = []
        story = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 行首是数字，后面跟着空格
                nid, line = line.split(" ", 1)
                nid = int(nid)
                
                # 新故事开始
                if nid == 1 and story:
                    stories.append(self._process_story(story))
                    story = []
                
                # 检查是否是问题行
                if "\t" in line:
                    question, answer, supporting = line.split("\t")
                    # 支持事实编号
                    supporting = list(map(int, supporting.split()))
                    story.append({
                        "id": nid, 
                        "type": "question", 
                        "text": question, 
                        "answer": answer,
                        "supporting": supporting
                    })
                else:
                    # 陈述句
                    story.append({"id": nid, "type": "statement", "text": line})
        
        # 添加最后一个故事
        if story:
            stories.append(self._process_story(story))
        
        logger.debug(f"解析完成，共{len(stories)}个故事")
        return stories
    
    def _process_story(self, story_lines: List[Dict]) -> Dict:
        """
        处理一个故事的行数据，整合为一个故事对象
        
        Args:
            story_lines: 故事的行数据
            
        Returns:
            Dict: 处理后的故事对象
        """
        statements = []
        questions = []
        
        for line in story_lines:
            if line["type"] == "statement":
                statements.append({
                    "id": line["id"],
                    "text": line["text"]
                })
            else:  # question
                questions.append({
                    "id": line["id"],
                    "text": line["text"],
                    "answer": line["answer"],
                    "supporting": line["supporting"]
                })
        
        return {
            "statements": statements,
            "questions": questions
        }
    
    def prepare_for_symbolic_engine(self, story: Dict) -> Tuple[List[str], List[str]]:
        """
        将故事数据转换为适合符号引擎处理的格式
        
        Args:
            story: 故事数据
            
        Returns:
            Tuple[List[str], List[str]]: (事实列表, 问题列表)
        """
        facts = []
        queries = []
        
        # 处理陈述句为事实
        for stmt in story["statements"]:
            facts.append(stmt["text"])
        
        # 处理问题
        for q in story["questions"]:
            # 找出支持性事实
            supporting_facts = []
            for fact_id in q["supporting"]:
                for stmt in story["statements"]:
                    if stmt["id"] == fact_id:
                        supporting_facts.append(stmt["text"])
                        break
            
            queries.append({
                "question": q["text"],
                "answer": q["answer"],
                "supporting_facts": supporting_facts
            })
        
        return facts, queries
    
    def get_all_tasks(self) -> List[int]:
        """返回所有可用任务的ID列表"""
        return list(range(1, 21))
    
    def get_task_name(self, task_id: int) -> str:
        """获取任务名称"""
        return BABI_TASKS.get(task_id, f"未知任务{task_id}")


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = BabiDataLoader()
    loader.download_dataset()
    
    # 加载任务1的训练数据
    task1_data = loader.load_task(1, "train")
    print(f"任务1({loader.get_task_name(1)})训练数据: {len(task1_data)}个故事")
    
    # 打印第一个故事的内容
    if task1_data:
        print("\n第一个故事示例:")
        story = task1_data[0]
        print("陈述句:")
        for stmt in story["statements"]:
            print(f"  {stmt['id']}: {stmt['text']}")
        
        print("\n问题:")
        for q in story["questions"]:
            print(f"  问题: {q['text']}")
            print(f"  答案: {q['answer']}")
            print(f"  支持事实ID: {q['supporting']}")
        
        # 转换为符号引擎格式
        facts, queries = loader.prepare_for_symbolic_engine(story)
        print("\n符号引擎格式:")
        print("事实:")
        for fact in facts:
            print(f"  {fact}")
        
        print("\n查询:")
        for query in queries:
            print(f"  问题: {query['question']}")
            print(f"  答案: {query['answer']}")
            print(f"  支持事实: {query['supporting_facts']}") 