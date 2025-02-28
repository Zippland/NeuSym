#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bAbI任务演示脚本

演示如何使用NeuSym框架处理bAbI推理任务。
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
from tqdm import tqdm

from neusym.data.babi_loader import BabiDataLoader
from neusym.core.symbolic_engine import SymbolicEngine
from neusym.core.neural_symbolic_interface import NeuralSymbolicInterface
from neusym.core.hybrid_reasoning_controller import HybridReasoningController
from neusym.core.babi_symbolic_adapter import BabiSymbolicAdapter
from neusym.models.language_model import MockLanguageModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("babi_demo")

def setup_system(model_name="mock", device="cpu", use_adapter=False, task_id=None):
    """
    设置NeuSym系统
    
    Args:
        model_name: 语言模型名称
        device: 运行设备
        use_adapter: 是否使用bAbI专用符号适配器
        task_id: 如果使用适配器，指定任务ID
        
    Returns:
        controller: 混合推理控制器实例
    """
    logger.info(f"正在初始化系统 (模型: {model_name}, 设备: {device}, 适配器: {use_adapter})")
    
    # 初始化语言模型
    if model_name.lower() == "mock":
        logger.info("使用模拟语言模型")
        lm = MockLanguageModel()
    else:
        try:
            from neusym.models.language_model import LanguageModel
            lm = LanguageModel(model_name_or_path=model_name, device=device)
            logger.info(f"语言模型加载成功: {model_name}")
        except Exception as e:
            logger.error(f"加载语言模型时出错: {e}")
            logger.warning("使用模拟模型进行演示")
            lm = MockLanguageModel()
    
    # 初始化符号引擎
    symbolic_engine = SymbolicEngine()
    
    # 初始化神经-符号接口
    interface = NeuralSymbolicInterface(lm)
    
    # 初始化符号适配器（如果需要）
    if use_adapter:
        adapter = BabiSymbolicAdapter(symbolic_engine)
        if task_id is not None:
            adapter.configure_for_task(task_id)
            logger.info(f"已为任务{task_id}配置符号适配器")
    
    # 初始化混合推理控制器
    controller = HybridReasoningController(
        language_model=lm,
        symbolic_engine=symbolic_engine,
        neural_symbolic_interface=interface
    )
    
    return controller

def run_single_example(controller, facts, query, use_adapter=False, adapter=None):
    """
    运行单个示例
    
    Args:
        controller: 混合推理控制器
        facts: 事实列表
        query: 查询信息
        use_adapter: 是否使用bAbI适配器
        adapter: bAbI符号适配器实例
        
    Returns:
        result: 推理结果
        correct: 是否正确
    """
    # 将事实添加到系统
    if use_adapter and adapter:
        # 使用适配器解析和添加事实
        for fact in facts:
            parsed_facts = adapter.extract_predicates_from_text(fact)
            for parsed_fact in parsed_facts:
                controller.symbolic_engine.add_fact(parsed_fact)
    else:
        # 使用神经符号接口解析和添加事实
        for fact in facts:
            controller.add_fact(fact)
    
    # 处理问题
    question = query["question"]
    expected_answer = query["answer"]
    
    # 使用控制器处理查询
    if use_adapter and adapter:
        # 使用适配器处理问题
        result = adapter.process_question(question)
    else:
        # 使用控制器处理查询
        result = controller.process_query(question)
    
    # 检查结果是否正确
    correct = result.lower() == expected_answer.lower()
    
    return result, correct

def run_task_evaluation(task_id, num_examples=10, model_name="mock", device="cpu", use_adapter=False):
    """
    评估系统在特定bAbI任务上的表现
    
    Args:
        task_id: 任务ID (1-20)
        num_examples: 评估样本数
        model_name: 语言模型名称
        device: 运行设备
        use_adapter: 是否使用bAbI专用符号适配器
        
    Returns:
        accuracy: 准确率
        results: 详细结果
    """
    # 初始化系统
    controller = setup_system(model_name, device, use_adapter, task_id)
    
    # 如果使用适配器，初始化bAbI适配器
    adapter = None
    if use_adapter:
        adapter = BabiSymbolicAdapter(controller.symbolic_engine)
        adapter.configure_for_task(task_id)
    
    # 加载数据
    loader = BabiDataLoader()
    task_data = loader.load_task(task_id, "test")
    
    # 限制样本数
    if num_examples > 0:
        task_data = task_data[:num_examples]
    
    logger.info(f"开始评估任务{task_id} ({loader.get_task_name(task_id)}), {len(task_data)}个样本")
    
    results = []
    correct_count = 0
    total_count = 0
    
    for i, story in enumerate(tqdm(task_data)):
        # 预处理故事数据
        facts, queries = loader.prepare_for_symbolic_engine(story)
        
        story_results = []
        
        for query in queries:
            # 重置引擎状态
            controller.reset()
            
            # 运行示例
            try:
                result, correct = run_single_example(
                    controller, facts, query, 
                    use_adapter=use_adapter, 
                    adapter=adapter
                )
                
                if correct:
                    correct_count += 1
                total_count += 1
                
                story_results.append({
                    "question": query["question"],
                    "expected": query["answer"],
                    "predicted": result,
                    "correct": correct
                })
                
            except Exception as e:
                logger.error(f"处理样本{i}时出错: {e}")
                story_results.append({
                    "question": query["question"],
                    "expected": query["answer"],
                    "predicted": "错误",
                    "correct": False,
                    "error": str(e)
                })
                total_count += 1
        
        results.append({
            "id": i,
            "facts": facts,
            "queries": story_results
        })
    
    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    logger.info(f"任务{task_id}评估完成, 准确率: {accuracy:.2f} ({correct_count}/{total_count})")
    
    return accuracy, results

def demo_interactive(model_name="mock", device="cpu", use_adapter=False):
    """交互式演示"""
    controller = setup_system(model_name, device, use_adapter)
    
    # 初始化bAbI适配器（如果需要）
    adapter = None
    if use_adapter:
        adapter = BabiSymbolicAdapter(controller.symbolic_engine)
        # 默认配置任务1
        adapter.configure_for_task(1)
    
    print("\n=== NeuSym交互式演示 ===")
    print("输入'退出'或'exit'结束对话。")
    print("输入'加载<任务ID>'加载bAbI任务示例。")
    if use_adapter:
        print("使用bAbI符号适配器进行推理。")
    
    facts = []
    current_task = 1
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["退出", "exit", "quit"]:
                print("感谢使用，再见！")
                break
            
            # 处理加载任务命令
            if user_input.startswith("加载") or user_input.startswith("load"):
                try:
                    task_id = int(user_input.split()[1])
                    loader = BabiDataLoader()
                    task_data = loader.load_task(task_id, "train")
                    
                    # 更新当前任务
                    current_task = task_id
                    
                    # 如果使用适配器，为新任务配置适配器
                    if use_adapter and adapter:
                        adapter.configure_for_task(task_id)
                    
                    if task_data:
                        story = task_data[0]
                        facts, queries = loader.prepare_for_symbolic_engine(story)
                        
                        print(f"\n已加载任务{task_id} ({loader.get_task_name(task_id)})的示例:")
                        print("事实:")
                        for i, fact in enumerate(facts):
                            print(f"  {i+1}. {fact}")
                        
                        print("\n问题示例:")
                        for i, query in enumerate(queries):
                            print(f"  {i+1}. {query['question']} (答案: {query['answer']})")
                        
                        # 重置控制器
                        controller.reset()
                        
                        # 添加事实到系统
                        if use_adapter and adapter:
                            for fact in facts:
                                parsed_facts = adapter.extract_predicates_from_text(fact)
                                for parsed_fact in parsed_facts:
                                    controller.symbolic_engine.add_fact(parsed_fact)
                        else:
                            for fact in facts:
                                controller.add_fact(fact)
                        
                        print("\n事实已加载到推理引擎。可以开始提问。")
                    else:
                        print(f"没有找到任务{task_id}的数据")
                
                except Exception as e:
                    print(f"加载任务时出错: {e}")
                
                continue
            
            # 处理添加事实命令
            if user_input.startswith("添加") or user_input.startswith("add"):
                fact = user_input.split(" ", 1)[1]
                facts.append(fact)
                
                # 添加到引擎
                if use_adapter and adapter:
                    parsed_facts = adapter.extract_predicates_from_text(fact)
                    for parsed_fact in parsed_facts:
                        controller.symbolic_engine.add_fact(parsed_fact)
                    print(f"已添加事实: {fact} (解析为: {', '.join(str(f) for f in parsed_facts)})")
                else:
                    success = controller.add_fact(fact)
                    if success:
                        print(f"已添加事实: {fact}")
                    else:
                        print(f"添加事实失败: {fact}")
                continue
            
            # 处理显示事实命令
            if user_input.startswith("事实") or user_input.startswith("facts"):
                print("\n当前事实:")
                for i, fact in enumerate(facts):
                    print(f"  {i+1}. {fact}")
                continue
            
            # 处理重置命令
            if user_input.startswith("重置") or user_input.startswith("reset"):
                facts = []
                controller.reset()
                if use_adapter and adapter:
                    adapter.configure_for_task(current_task)
                print("系统已重置")
                continue
            
            # 处理普通问题
            if use_adapter and adapter:
                result = adapter.process_question(user_input)
            else:
                result = controller.process_query(user_input)
            
            print(f"\n{result}")
            
            # 显示推理过程
            if not use_adapter:
                trace = controller.get_reasoning_trace()
                if trace:
                    print("\n推理过程:")
                    for step in trace:
                        print(f"  {step}")
            
        except KeyboardInterrupt:
            print("\n程序被用户中断。再见！")
            break
        except Exception as e:
            logger.error(f"处理输入时出错: {e}")
            print(f"处理您的问题时出错了: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="bAbI任务演示")
    
    parser.add_argument("--task", type=int, default=0,
                        help="要评估的bAbI任务ID (1-20)，0表示交互式模式")
    
    parser.add_argument("--model", type=str, default="mock",
                        help="使用的语言模型，'mock'表示使用模拟模型")
    
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="运行设备")
    
    parser.add_argument("--examples", type=int, default=10,
                        help="评估的示例数量")
    
    parser.add_argument("--use_adapter", action="store_true",
                        help="是否使用bAbI特定的符号适配器")
    
    args = parser.parse_args()
    
    if args.task == 0:
        # 交互式模式
        demo_interactive(
            model_name=args.model,
            device=args.device,
            use_adapter=args.use_adapter
        )
    else:
        # 评估特定任务
        accuracy, results = run_task_evaluation(
            task_id=args.task,
            num_examples=args.examples,
            model_name=args.model,
            device=args.device,
            use_adapter=args.use_adapter
        )
        
        # 显示一些示例结果
        print("\n示例结果:")
        for i, story in enumerate(results[:3]):  # 仅显示前3个故事
            print(f"\n故事 {i+1}:")
            
            print("事实:")
            for j, fact in enumerate(story["facts"][:5]):  # 最多显示5个事实
                print(f"  {j+1}. {fact}")
            
            if len(story["facts"]) > 5:
                print(f"  ... 还有{len(story['facts'])-5}个事实")
            
            print("\n问题和回答:")
            for j, query in enumerate(story["queries"]):
                status = "✓" if query["correct"] else "✗"
                print(f"  {j+1}. 问题: {query['question']}")
                print(f"     预期答案: {query['expected']}")
                print(f"     系统回答: {query['predicted']} {status}")
        
        print(f"\n总准确率: {accuracy:.2f}")

if __name__ == "__main__":
    main() 