# SortWise - 智能垃圾分类系统

基于深度学习的智能垃圾分类系统，可以准确识别和分类不同类型的垃圾。

## 功能特点

- 支持6种垃圾类别的识别：可回收物、有害垃圾、厨余垃圾、其他垃圾、大件垃圾、装修垃圾
- 使用EfficientNet作为基础模型，提供高准确率的分类结果
- 提供Web界面，方便用户上传图片进行识别
- 支持批量处理和实时预测
- 提供详细的评估指标和可视化结果

## 项目结构

```
sortwise/
├── app/                    # Web应用程序
│   ├── api/               # API路由和处理
│   ├── static/           # 静态文件
│   └── templates/        # HTML模板
├── ml/                    # 机器学习模块
│   ├── models/          # 模型定义
│   ├── data/            # 数据处理
│   ├── train/           # 训练相关
│   └── utils/           # 工具函数
├── data/                 # 数据目录
├── scripts/              # 实用脚本
├── tests/               # 测试目录
├── notebooks/           # Jupyter notebooks
├── logs/                # 日志文件
├── checkpoints/         # 模型检查点
└── docs/                # 文档
```

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/sortwise.git
cd sortwise
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -e .
```

## 使用方法

### 训练模型

```bash
python scripts/train.py --config config.yaml
```

### 评估模型

```bash
python scripts/evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth --data-dir data/test
```

### 预测单张图片

```bash
python scripts/predict.py --config config.yaml --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
```

### 启动Web应用

```bash
python app/run.py
```

## 配置

所有配置都在 `config.yaml` 文件中，包括：

- 模型配置（架构、类别数等）
- 训练配置（批次大小、学习率等）
- 数据配置（数据增强、归一化等）
- 应用配置（主机、端口等）

## 开发

1. 安装开发依赖：
```bash
pip install -e ".[dev]"
```

2. 运行测试：
```bash
pytest tests/
```

3. 代码格式化：
```bash
black .
isort .
```

## 贡献

欢迎提交 Pull Request 或创建 Issue！

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 作者

[您的姓名] - [您的学校]

## 致谢

- 感谢指导教师的指导
- 感谢开源社区提供的工具和框架

#   S o r t w i s e  
 