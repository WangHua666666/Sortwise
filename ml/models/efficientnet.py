import os
import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
from typing import List, Tuple, Optional, Union

class WasteClassifier(nn.Module):
    """
    废弃物分类模型类
    
    基于 EfficientNet 的迁移学习模型，用于废弃物分类任务。
    模型结构：
    1. EfficientNet主干网络（预训练）
    2. 特征提取层
    3. Dropout层（防止过拟合）
    4. 全连接分类头
    """
    
    def __init__(self, 
                 num_classes: int = 6, 
                 model_name: str = 'efficientnet_b0', 
                 pretrained: bool = True,
                 dropout_rate: float = 0.2):
        """
        初始化模型
        
        Args:
            num_classes: 分类类别数，默认为6类废弃物
            model_name: EfficientNet 模型版本，可选 b0-b7
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout层的丢弃率，用于防止过拟合
        """
        super(WasteClassifier, self).__init__()
        
        # 加载基础模型
        self.base_model = timm.create_model(model_name, pretrained=pretrained)
        
        # 获取特征维度
        if 'efficientnet' in model_name:
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
            
        # 添加自定义分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x: 输入图像张量，形状为 (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: 模型输出的类别logits
        """
        # 特征提取
        features = self.base_model(x)
        
        # 分类头
        x = self.dropout(features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def get_top5_predictions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 TOP5 预测结果
        
        Args:
            x: 输入图像张量，形状为 (batch_size, 3, height, width)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (TOP5概率值, TOP5类别索引)
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            top5_prob, top5_indices = torch.topk(probabilities, k=5, dim=1)
            return top5_prob, top5_indices
    
    def predict(self, x: torch.Tensor) -> Tuple[List[str], List[float]]:
        """
        预测单张图片的类别和置信度
        
        Args:
            x: 输入图像张量，形状为 (1, 3, height, width)
            
        Returns:
            Tuple[List[str], List[float]]: (预测类别名称列表, 置信度列表)
        """
        top5_prob, top5_indices = self.get_top5_predictions(x)
        
        class_names = self.get_class_names()
        predicted_classes = [class_names[idx] for idx in top5_indices[0].tolist()]
        confidences = top5_prob[0].tolist()
        
        return predicted_classes, confidences
    
    @staticmethod
    def get_class_names() -> List[str]:
        """
        获取废弃物分类的类别名称
        
        Returns:
            List[str]: 包含所有类别名称的列表
        """
        return ['可回收物', '有害垃圾', '厨余垃圾', '其他垃圾', '大件垃圾', '装修垃圾']

def get_model(num_classes: int = 6,
             weights_path: Optional[str] = None,
             device: str = 'cpu') -> WasteClassifier:
    """
    获取预训练的模型实例
    
    Args:
        num_classes: 分类类别数，默认为6类废弃物
        weights_path: 预训练权重文件路径，默认为None
        device: 运行设备，默认为'cpu'
    
    Returns:
        WasteClassifier: 配置好的模型实例
    """
    # 获取模型配置
    model_name = os.getenv('MODEL_NAME', 'efficientnet_b0')
    
    # 创建模型实例
    model = WasteClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=True
    )
    
    # 加载训练好的权重（如果提供）
    if weights_path and os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"成功加载模型权重: {weights_path}")
        except Exception as e:
            print(f"加载模型权重时出错: {str(e)}")
    
    # 设置为评估模式
    model.eval()
    return model 