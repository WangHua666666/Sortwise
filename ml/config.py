import torch
from pathlib import Path
from typing import Dict, List
import yaml

def load_config():
    """从YAML文件加载配置"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 加载配置
Config = load_config()

class Config:
    """
    项目配置类
    用于集中管理项目的所有配置参数，包括：
    - 文件路径配置
    - 模型参数配置
    - 训练参数配置
    - 数据处理配置
    """
    
    # 路径配置：设置项目中各种文件的存储位置
    ROOT_DIR = Path(__file__).parent.parent  # 项目根目录
    DATA_DIR = ROOT_DIR / 'data'  # 数据集存储目录
    MODEL_WEIGHTS_DIR = ROOT_DIR / 'checkpoints'  # 模型权重保存目录
    LOGS_DIR = ROOT_DIR / 'logs'  # 日志文件存储目录
    
    # 自动创建必要的目录
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # 模型配置：定义模型结构和参数
    MODEL_CONFIG = {
        'model_name': 'efficientnet_b0',  # 使用的模型架构
        'num_classes': 34,  # 分类类别数量
        'pretrained': False,  # 是否使用预训练权重
        'dropout_rate': 0.2,  # Dropout比率，用于防止过拟合
        'image_size': 224  # 输入图片尺寸
    }
    
    # 训练配置：定义训练过程的超参数
    TRAIN_CONFIG = {
        'batch_size': 32,  # 每批次训练的样本数，会根据GPU显存自动调整
        'num_epochs': 50,  # 总训练轮数
        'learning_rate': 0.001,  # 学习率
        'weight_decay': 1e-4,  # 权重衰减，用于正则化
        'early_stopping_patience': 5,  # 早停轮数：当验证集性能多少轮未提升时停止训练
        'num_workers': 2,  # 数据加载的进程数
        'pin_memory': True  # 是否将数据加载到CUDA固定内存，可加速GPU训练
    }
    
    # 数据配置：定义数据处理和加载的参数
    DATA_CONFIG = {
        'train_dir': str(DATA_DIR / 'train'),  # 训练集目录
        'val_dir': str(DATA_DIR / 'val'),  # 验证集目录
        'test_dir': str(DATA_DIR / 'test'),  # 测试集目录
        'image_size': MODEL_CONFIG['image_size'],  # 图片尺寸，与模型输入保持一致
        'mean': [0.485, 0.456, 0.406],  # 图片标准化的均值（使用ImageNet标准）
        'std': [0.229, 0.224, 0.225]  # 图片标准化的标准差（使用ImageNet标准）
    }
    
    # 类别信息：在训练时会根据实际数据集自动更新
    CLASSES = []  # 类别名称列表
    CLASS_TO_IDX = {}  # 类别名称到索引的映射字典
    
    # 设备配置：自动选择使用GPU还是CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def get_best_model_path(cls) -> Path:
        """获取最佳模型权重的保存路径"""
        return cls.MODEL_WEIGHTS_DIR / 'best_model.pth'
    
    @classmethod
    def get_latest_model_path(cls) -> Path:
        """获取最新一次保存的模型权重路径"""
        return cls.MODEL_WEIGHTS_DIR / 'latest_model.pth'
    
    @classmethod
    def get_interrupted_model_path(cls) -> Path:
        """获取训练中断时保存的模型权重路径"""
        return cls.MODEL_WEIGHTS_DIR / 'interrupted_model.pth'
    
    @classmethod
    def adjust_batch_size(cls) -> int:
        """
        根据GPU显存大小自动调整批次大小
        - 显存<6GB: 使用较小的batch_size=32
        - 显存<8GB: 使用中等的batch_size=48
        - 显存>=8GB: 使用配置文件中设定的batch_size
        """
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # 转换为GB
            if gpu_mem < 6:  # 小于6GB显存
                return 32
            elif gpu_mem < 8:  # 小于8GB显存
                return 48
            else:
                return cls.TRAIN_CONFIG['batch_size']
        return 32  # CPU模式下使用较小的batch_size 