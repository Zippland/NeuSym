#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NeuSym 快速入门示例

本脚本演示了如何使用NeuSym神经符号推理系统进行简单的推理任务。
"""

import logging
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

print("# NeuSym 快速入门")
print("本示例演示了如何使用NeuSym神经符号推理系统进行简单的推理任务。\n")

print("## 1. 初始化系统组件")
print("首先，我们初始化NeuSym的核心组件...\n")

from neusym.models.language_model import LanguageModel
from neusym.core.symbolic_engine import SymbolicEngine
from neusym.core.neural_symbolic_interface import NeuralSymbolicInterface
from neusym.core.hybrid_reasoning_controller import HybridReasoningController

# 初始化语言模型
print("正在初始化语言模型（这可能需要一些时间）...")
lm = LanguageModel(
    model_name_or_path="baichuan-inc/Baichuan-7B",  # 可以替换为其他小型模型
    device="cpu",  # 如果有GPU，可以设置为"cuda"
    load_in_4bit=False  # 如果有GPU，可以设置为True以节省显存
)

# 初始化符号推理引擎
symbolic_engine = SymbolicEngine()

# 初始化神经符号接口
interface = NeuralSymbolicInterface(lm)

# 初始化混合推理控制器
controller = HybridReasoningController(lm, symbolic_engine, interface)
print("系统组件初始化完成！\n")

print("## 2. 添加知识")
print("现在，我们向系统添加一些事实和规则...\n")

# 添加一些基本事实
facts_text = "约翰是一名医生，他住在纽约。玛丽是一名教师，她住在波士顿。"
print(f"添加事实: {facts_text}")
facts_result = controller.process_input(facts_text)

print(f"提取了 {facts_result['facts_count']} 个事实：")
for fact in facts_result['facts']:
    print(f"- {fact}")

# 添加一些规则
rules_text = "如果一个人是医生，那么他有医学学位。如果一个人住在纽约，那么他住在美国。"
print(f"\n添加规则: {rules_text}")
rules_result = controller.process_input(rules_text)

print(f"提取了 {rules_result['rules_count']} 个规则：")
for rule in rules_result['rules']:
    print(f"- {rule}")

print("\n## 3. 进行推理")
print("现在，我们可以向系统提问，测试其推理能力...\n")

# 提问：约翰有医学学位吗？
question1 = "约翰有医学学位吗？"
print(f"问题1: {question1}")
result1 = controller.answer_question(question1)

print(f"回答: {result1['answer']}")
print(f"推理路径: {result1['reasoning_path']}")
print(f"置信度: {result1['confidence']:.2f}")

if 'explanation' in result1 and result1['explanation']:
    print(f"\n解释:\n{result1['explanation']}")

# 提问：约翰住在美国吗？
question2 = "约翰住在美国吗？"
print(f"\n问题2: {question2}")
result2 = controller.answer_question(question2)

print(f"回答: {result2['answer']}")
print(f"推理路径: {result2['reasoning_path']}")
print(f"置信度: {result2['confidence']:.2f}")

if 'explanation' in result2 and result2['explanation']:
    print(f"\n解释:\n{result2['explanation']}")

# 提问：玛丽有医学学位吗？（这应该是否定的，因为玛丽是教师，不是医生）
question3 = "玛丽有医学学位吗？"
print(f"\n问题3: {question3}")
result3 = controller.answer_question(question3)

print(f"回答: {result3['answer']}")
print(f"推理路径: {result3['reasoning_path']}")
print(f"置信度: {result3['confidence']:.2f}")

if 'explanation' in result3 and result3['explanation']:
    print(f"\n解释:\n{result3['explanation']}")

print("\n## 4. 查看推理过程")
print("我们可以查看系统的推理过程，了解它是如何得出结论的...\n")

# 获取推理过程
reasoning_trace = controller.get_reasoning_trace()

print("推理过程:")
for step_type, step_content in reasoning_trace:
    if step_content:
        print(f"- {step_type}: {step_content}")
    else:
        print(f"- {step_type}")

print("\n## 5. 重置系统")
print("如果需要，我们可以重置系统，清空知识库和推理记录...\n")

# 重置系统
controller.reset()
print("系统已重置，知识库和推理记录已清空。")

print("\n## 6. 尝试更复杂的推理")
print("现在，让我们尝试一个更复杂的推理场景...\n")

# 添加更复杂的知识
complex_facts = """
苏格拉底是一个人。
所有的人都是凡人。
凡人终有一死。
"""

print(f"添加知识: {complex_facts}")
controller.process_input(complex_facts)

# 提问
complex_question = "苏格拉底会死吗？"
print(f"\n问题: {complex_question}")
complex_result = controller.answer_question(complex_question)

print(f"回答: {complex_result['answer']}")
print(f"推理路径: {complex_result['reasoning_path']}")
print(f"置信度: {complex_result['confidence']:.2f}")

if 'explanation' in complex_result and complex_result['explanation']:
    print(f"\n解释:\n{complex_result['explanation']}")

print("\n## 7. 清理资源")
print("最后，我们应该卸载语言模型以释放内存...\n")

# 卸载语言模型
lm.unload()
print("语言模型已卸载，内存已释放。")
print("\n示例结束！") 