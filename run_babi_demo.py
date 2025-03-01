#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bAbI任务演示执行脚本

用于从命令行直接运行bAbI任务演示。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

if __name__ == "__main__":
    # 导入演示模块
    from src.notebooks.babi_demo import main
    
    # 运行演示（传递命令行参数）
    main() 