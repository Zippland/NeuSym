#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NeuSym演示脚本

提供一个简单的命令行界面，用于演示神经符号推理系统的功能。
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from src.models.language_model import LanguageModel
from src.core.symbolic_engine import SymbolicEngine
from src.core.neural_symbolic_interface import NeuralSymbolicInterface
from src.core.hybrid_reasoning_controller import HybridReasoningController

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("neusym.demo")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="NeuSym演示脚本")
    
    parser.add_argument("--model", type=str, default="baichuan-inc/Baichuan-7B",
                        help="使用的基础语言模型")
    
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="运行设备")
    
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="是否使用4位量化加载模型")
    
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="是否使用8位量化加载模型")
    
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细日志")
    
    return parser.parse_args()

def initialize_system(args):
    """初始化系统组件"""
    logger.info("正在初始化系统组件...")
    
    # 初始化语言模型
    lm = LanguageModel(
        model_name_or_path=args.model,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )
    
    # 初始化符号推理引擎
    symbolic_engine = SymbolicEngine()
    
    # 初始化神经符号接口
    interface = NeuralSymbolicInterface(lm)
    
    # 初始化混合推理控制器
    controller = HybridReasoningController(lm, symbolic_engine, interface)
    
    logger.info("系统初始化完成")
    return controller

def run_interactive_demo(controller):
    """运行交互式演示"""
    print("\n欢迎使用NeuSym神经符号推理系统！")
    print("输入'退出'或'exit'结束对话，输入'重置'或'reset'重置知识库。")
    print("输入'帮助'或'help'查看更多命令。")
    print("\n提示: 先输入一些事实和规则，然后提问。例如:")
    print("  > 约翰是一名医生，他住在纽约。")
    print("  > 如果一个人是医生，那么他有医学学位。")
    print("  > 约翰有医学学位吗？\n")
    
    while True:
        try:
            user_input = input("\n> ")
            
            # 检查特殊命令
            if user_input.lower() in ["退出", "exit", "quit"]:
                print("感谢使用，再见！")
                break
            elif user_input.lower() in ["重置", "reset"]:
                controller.reset()
                print("知识库已重置。")
                continue
            elif user_input.lower() in ["帮助", "help"]:
                show_help()
                continue
            elif user_input.lower() in ["跟踪", "trace"]:
                show_reasoning_trace(controller)
                continue
            
            # 判断是否为问题
            if is_question(user_input):
                # 处理问题
                result = controller.answer_question(user_input)
                print(f"\n回答: {result['answer']}")
                print(f"推理路径: {result['reasoning_path']}")
                print(f"置信度: {result['confidence']:.2f}")
                
                if 'explanation' in result and result['explanation']:
                    print(f"\n解释: {result['explanation']}")
            else:
                # 处理陈述句，提取事实和规则
                result = controller.process_input(user_input)
                print(f"已提取 {result['facts_count']} 个事实和 {result['rules_count']} 个规则。")
                
                # 显示提取的内容
                if result['facts_count'] > 0:
                    print("\n提取的事实:")
                    for fact in result['facts']:
                        print(f"- {fact}")
                
                if result['rules_count'] > 0:
                    print("\n提取的规则:")
                    for rule in result['rules']:
                        print(f"- {rule}")
        
        except KeyboardInterrupt:
            print("\n程序被用户中断。再见！")
            break
        except Exception as e:
            logger.exception(f"处理输入时出错: {e}")
            print(f"处理您的输入时出错了: {str(e)}")

def is_question(text):
    """简单判断文本是否为问题"""
    question_markers = ["?", "？", "吗", "谁", "什么", "哪", "怎么", "如何", "为什么", "何时", "何地"]
    for marker in question_markers:
        if marker in text:
            return True
    return False

def show_help():
    """显示帮助信息"""
    print("\n可用命令:")
    print("  退出, exit, quit - 结束程序")
    print("  重置, reset - 清空知识库")
    print("  跟踪, trace - 显示最近的推理过程")
    print("  帮助, help - 显示此帮助信息")
    print("\n使用提示:")
    print("  1. 先输入一些事实和规则，例如:")
    print("     > 约翰是一名医生，他住在纽约。")
    print("     > 如果一个人是医生，那么他有医学学位。")
    print("  2. 然后提问，例如:")
    print("     > 约翰有医学学位吗？")
    print("  3. 系统会根据已知信息进行推理并回答")

def show_reasoning_trace(controller):
    """显示推理过程"""
    trace = controller.get_reasoning_trace()
    if not trace:
        print("尚无推理过程记录。")
        return
    
    print("\n推理过程:")
    for step_type, step_content in trace:
        if step_content:
            print(f"- {step_type}: {step_content}")
        else:
            print(f"- {step_type}")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger("neusym").setLevel(logging.DEBUG)
    
    try:
        # 初始化系统
        controller = initialize_system(args)
        
        # 运行交互式演示
        run_interactive_demo(controller)
        
        return 0
    
    except Exception as e:
        logger.exception(f"程序运行出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 