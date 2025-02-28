#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import re

# 读取版本号
with open(os.path.join("neusym", "__init__.py"), "r", encoding="utf-8") as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("无法找到版本信息")

# 读取README作为长描述
with open("neusym/README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# 读取依赖
with open("neusym/requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="neusym",
    version=version,
    description="神经符号推理增强的小型语言模型",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NeuSym Team",
    author_email="example@example.com",
    url="https://github.com/yourusername/neusym",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "neusym=neusym.main:main",
            "neusym-demo=neusym.demo:main",
        ],
    },
) 