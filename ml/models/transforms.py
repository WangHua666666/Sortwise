import torchvision.transforms as T

class ImageTransform:
    """图像预处理"""
    
    def __init__(self):
        # 基础预处理
        base_transforms = [
            T.Resize((224, 224)),  # 统一大小
            T.ToTensor(),  # 转换为张量
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ]
        
        # 训练时的预处理
        self.train_transform = T.Compose([
            *base_transforms[:-1],  # 除了标准化之前的所有步骤
            T.RandomHorizontalFlip(),  # 随机水平翻转
            base_transforms[-1]  # 标准化
        ])
        
        # 验证和预测时的预处理
        self.val_transform = T.Compose(base_transforms)
    
    def process_image(self, image):
        """处理单张图片"""
        tensor = self.val_transform(image)
        return tensor.unsqueeze(0)
    
    @staticmethod
    def load_image(path):
        """加载图片"""
        return Image.open(path).convert('RGB') 