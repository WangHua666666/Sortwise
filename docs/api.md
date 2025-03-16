# SortWise API 文档

## REST API

### 图像分类

**端点**: `/api/predict`

**方法**: POST

**描述**: 上传图片并获取垃圾分类预测结果

**请求**:
- Content-Type: multipart/form-data
- Body: 
  - image: 图片文件 (支持 jpg, jpeg, png)

**响应**:
```json
{
    "success": true,
    "prediction": {
        "class": "可回收物",
        "confidence": 0.95,
        "top_predictions": [
            {"class": "可回收物", "confidence": 0.95},
            {"class": "其他垃圾", "confidence": 0.03},
            {"class": "厨余垃圾", "confidence": 0.02}
        ]
    }
}
```

### 获取支持的分类

**端点**: `/api/categories`

**方法**: GET

**描述**: 获取系统支持的垃圾分类类别

**响应**:
```json
{
    "success": true,
    "categories": [
        "可回收物",
        "有害垃圾",
        "厨余垃圾",
        "其他垃圾",
        "大件垃圾",
        "装修垃圾"
    ]
}
```

## Python API

### WasteClassifier

```python
from ml.models.classifier import WasteClassifier

# 创建模型实例
model = WasteClassifier(
    model_name="efficientnet-b0",
    num_classes=6,
    pretrained=False
)

# 预测
predictions = model.predict(image)
```

### WasteDataset

```python
from ml.data.dataset import WasteDataset

# 创建数据集
dataset = WasteDataset(
    data_dir="data/processed",
    transform=transforms
)
```

### Trainer

```python
from ml.train.trainer import Trainer

# 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)

# 训练模型
trainer.train(num_epochs=100)
``` 