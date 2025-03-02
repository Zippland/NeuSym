# NeuSym: 让小型AI更聪明的神经符号推理系统

NeuSym是一个创新性的神经符号融合框架，通过将小型语言模型的自然语言处理能力与符号推理系统的精确逻辑能力结合，让体积小、资源需求低的AI模型也能进行复杂的逻辑推理。

## 核心思想：结合两种思维方式

想象一下，人类的思维有两种模式：

1. **直觉思维**：快速、灵活，但可能不精确（比如看到一只动物立刻认出是猫）
2. **逻辑思维**：慢一些，但严谨精确（比如解数学题时一步步推导）

NeuSym就是把这两种思维方式结合起来：
- **神经网络部分**（就像AI的"直觉"）：理解自然语言，灵活处理文本
- **符号推理部分**（就像AI的"逻辑"）：严格按规则进行推理，确保结论正确

## 为什么需要这个项目？

现在的大型AI模型（如ChatGPT）虽然强大，但需要庞大的计算资源。而小型AI模型虽然资源需求低，但在复杂推理时常常出错。NeuSym就像给小型AI装了一个"逻辑助手"，让它也能处理复杂的推理问题，而且不需要特别强大的电脑。

[查看完整项目计划](plan.md)

## 项目特点

- 专为小型语言模型（1-3B参数级别）设计的神经符号架构
- 高效的神经-符号转换机制，实现自然语言与逻辑表示的双向映射
- 基于不确定性的混合推理控制器，智能决策处理路径
- 可扩展的领域知识编码框架，支持不同专业领域的快速适配
- 资源自适应的系统架构，优化在有限计算资源下的性能

## 当前进展

框架已完成核心功能模块的开发，包括：

- ✅ 符号推理引擎：支持谓词逻辑表示和基础推理
- ✅ 神经-符号接口：实现了自然语言与逻辑形式的双向转换
- ✅ 混合推理控制器：智能决策使用哪种推理路径
- ✅ bAbI任务适配器：针对推理任务的特定转换
- ✅ 数据加载和处理：支持bAbI数据集的加载和处理
- ✅ 基础演示界面：交互式测试bAbI任务

## 项目结构

```
neusym/
├── core/            # 核心推理引擎和接口
│   ├── symbolic_engine.py         # 符号推理引擎
│   ├── neural_symbolic_interface.py # 神经-符号转换接口
│   ├── hybrid_reasoning_controller.py # 混合推理控制器
│   └── babi_symbolic_adapter.py   # bAbI任务适配器
├── models/          # 语言模型相关代码
│   └── language_model.py          # 语言模型封装
├── utils/           # 工具函数和辅助模块
├── data/            # 数据集和示例
│   └── babi_loader.py             # bAbI数据集加载器
├── tests/           # 测试代码
│   ├── test_basic.py              # 基础功能测试
│   └── test_babi_loader.py        # 数据加载器测试
├── notebooks/       # 笔记本示例
│   ├── quickstart.py              # 快速入门示例
│   └── babi_demo.py               # bAbI任务演示
└── docs/            # 文档
```

## 环境要求

- Python 3.9+
- PyTorch 1.13+
- 8GB+ RAM

## 安装方法

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/neusym.git
cd neusym
```

### 2. 安装依赖

```bash
pip install -r src/requirements.txt
```

### 3. 安装开发模式

```bash
pip install -e .
```

## 使用方法

### 运行bAbI任务演示

bAbI任务是Facebook AI Research设计的20个评估AI系统推理能力的任务集合。NeuSym框架支持在这些任务上的测试和评估。

#### 交互式演示

```bash
python run_babi_demo.py
```

或者使用内置命令：

```bash
neusym-demo
```

#### 评估特定任务

```bash
python run_babi_demo.py --task 1 --examples 50
```

其中：
- `--task`：指定要评估的bAbI任务ID (1-20)
- `--examples`：指定要评估的样本数量
- `--model`：指定使用的语言模型，默认为"baichuan-1b"
- `--device`：指定运行设备，可选项为"cpu"、"cuda"或"auto"

### 使用bAbI数据加载器

```python
from neusym.data.babi_loader import BabiDataLoader

# 初始化加载器
loader = BabiDataLoader()

# 下载数据集(首次使用时需要)
loader.download_dataset()

# 加载任务1的训练数据
task1_data = loader.load_task(1, "train")

# 获取适合符号引擎的数据格式
story = task1_data[0]
facts, queries = loader.prepare_for_symbolic_engine(story)

# 打印数据
print(f"事实数量: {len(facts)}")
print(f"查询数量: {len(queries)}")
```

### 使用符号推理引擎

```python
from neusym.core.symbolic_engine import SymbolicEngine

# 初始化推理引擎
engine = SymbolicEngine()

# 添加事实
engine.add_fact("At(Jhon, kitchen)")
engine.add_fact("Has(Jhon, apple)")

# 添加规则
engine.add_rule("At(X, Y) & has(X, Z) -> At(Z, Y)")

# 执行推理
results = engine.reason("At(apple, where)")
print(results)  # 输出: ["At(apple, kitchen)"]
```

## 测试

运行单元测试：

```bash
pytest -xvs neusym/tests/
```

测试特定模块：

```bash
pytest -xvs neusym/tests/test_babi_loader.py
```

## 下一步计划

- 支持更多的bAbI任务类型
- 优化混合推理控制器策略
- 提高系统在限制资源环境下的性能
- 增强推理过程的可视化

## 贡献指南

欢迎贡献代码和提出建议！请提交issue或pull request。

## 引用

如果您在研究或项目中使用了NeuSym，请引用我们的项目：

```
@misc{neusym2024,
  title={NeuSym: 神经符号推理增强的小型语言模型},
  author={Zippland},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/zippland/neusym}}
}
``` 