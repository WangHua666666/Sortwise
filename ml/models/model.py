import os
import torch
import torch.nn as nn
import torchvision.models as models

def get_model():
    """
    获取深度学习模型实例
    
    使用预训练的ResNet50模型，并根据我们的垃圾分类任务修改最后的全连接层。
    支持通过环境变量配置：
    - MODEL_ARCHITECTURE: 模型架构（默认：resnet50）
    - NUM_CLASSES: 分类类别数（默认：4）
    
    Returns:
        torch.nn.Module: 配置好的模型实例
        
    Raises:
        ValueError: 当指定了不支持的模型架构时
    """
    # 从环境变量获取配置
    model_arch = os.environ.get('MODEL_ARCHITECTURE', 'resnet50')
    num_classes = int(os.environ.get('NUM_CLASSES', 4))

    if model_arch == 'resnet50':
        # 加载预训练的ResNet50模型
        model = models.resnet50(pretrained=True)
        # 修改最后的全连接层以适应我们的分类任务
        # 原始的fc层输入特征维度不变，输出改为我们需要的类别数
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型架构: {model_arch}")

    return model 