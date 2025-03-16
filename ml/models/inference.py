import torch
import timm
from PIL import Image
from .transforms import ImageTransform

class Predictor:
    """图像分类预测器"""
    
    def __init__(self, model_path='ml/models/checkpoints/best_model.pth'):
        """初始化预测器"""
        # 创建模型
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
        
        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print("模型加载成功")
        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")
        
        self.model.eval()
        
        # 创建转换器
        self.transforms = ImageTransform()
        
        # 类别映射
        self.classes = ['可回收物', '有害垃圾', '厨余垃圾', '其他垃圾']
    
    def predict(self, image_path):
        """预测图片类别
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: {
                'category': 预测类别,
                'confidence': 置信度,
                'success': 是否成功
            }
        """
        try:
            # 加载并处理图片
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms.val_transform(image).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                prob, pred = torch.max(probs, dim=1)
                
                return {
                    'category': self.classes[pred.item()],
                    'confidence': float(prob.item()),
                    'success': True
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

if __name__ == '__main__':
    # 测试代码
    predictor = Predictor()
    result = predictor.predict('path_to_test_image.jpg')
    print(result) 