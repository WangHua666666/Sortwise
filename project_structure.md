# SortWise Project Structure

```
sortwise/
├── app/                    # Web应用程序
│   ├── __init__.py
│   ├── api/               # API路由和处理
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── static/           # 静态文件
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/        # HTML模板
│
├── ml/                    # 机器学习模块
│   ├── __init__.py
│   ├── config.py         # 配置管理
│   ├── models/          # 模型定义
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── data/            # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── train/           # 训练相关
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/           # 工具函数
│       ├── __init__.py
│       └── metrics.py
│
├── data/                 # 数据目录
│   ├── raw/             # 原始数据
│   ├── processed/       # 处理后的数据
│   └── external/        # 外部数据
│
├── scripts/              # 实用脚本
│   ├── train.py         # 训练脚本
│   ├── evaluate.py      # 评估脚本
│   └── predict.py       # 预测脚本
│
├── tests/               # 测试目录
│   ├── __init__.py
│   ├── test_models.py
│   └── test_data.py
│
├── notebooks/           # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── logs/                # 日志文件
│   ├── training/
│   └── app/
│
├── checkpoints/         # 模型检查点
│   └── models/
│
├── docs/                # 文档
│   ├── api.md
│   ├── setup.md
│   └── usage.md
│
├── .gitignore          # Git忽略文件
├── README.md           # 项目说明
├── requirements.txt    # 项目依赖
├── setup.py           # 安装脚本
└── config.yaml        # 全局配置文件
```

## 目录说明

### app/
Web应用程序相关的所有代码，包括API路由、静态文件和模板。

### ml/
机器学习相关的所有代码，包括模型定义、数据处理、训练逻辑等。

### data/
- raw/: 原始数据集
- processed/: 预处理后的数据
- external/: 外部数据源

### scripts/
独立的脚本文件，用于训练、评估和预测。

### tests/
单元测试和集成测试代码。

### notebooks/
用于数据分析和实验的Jupyter notebooks。

### logs/
应用程序和训练日志。

### checkpoints/
模型检查点和保存的模型文件。

### docs/
项目文档，包括API文档、设置指南和使用说明。

## 主要文件

- README.md: 项目概述和快速开始指南
- requirements.txt: Python依赖包列表
- setup.py: 项目安装脚本
- config.yaml: 全局配置文件

## 使用建议

1. 所有机器学习相关的代码都放在 `ml/` 目录下
2. Web应用相关的代码都放在 `app/` 目录下
3. 使用 `scripts/` 目录中的脚本进行训练和评估
4. 配置文件集中在根目录的 `config.yaml`
5. 所有实验性的代码都放在 `notebooks/` 目录 