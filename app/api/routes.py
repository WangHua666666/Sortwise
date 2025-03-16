import os
from flask import current_app, request, jsonify
from werkzeug.utils import secure_filename
from app.api import bp
from app.utils.model_utils import get_prediction
from app.utils.file_utils import allowed_file
from ml.config import Config

@bp.route('/predict', methods=['POST'])
def predict():
    """
    图像预测接口：接收上传的图片，返回废弃物分类预测结果
    
    请求方式：POST
    参数：
        file: 图片文件（multipart/form-data）
    
    返回：
        成功：
            {
                "success": true,
                "prediction": {
                    "category_id": 0,
                    "category_name": "可回收物",
                    "confidence": 95.5
                }
            }
        失败：
            {
                "error": "错误信息"
            }
    """
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 验证文件类型
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
    
    try:
        # 安全地获取文件名并保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 调用模型进行预测
        prediction = get_prediction(filepath)
        
        # 清理临时文件
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    
    except Exception as e:
        # 确保清理临时文件
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@bp.route('/categories', methods=['GET'])
def get_categories():
    """
    获取支持的废弃物分类类别
    
    请求方式：GET
    返回：
        {
            "categories": [
                "可回收物",
                "有害垃圾",
                "厨余垃圾",
                "其他垃圾",
                "大件垃圾",
                "装修垃圾"
            ],
            "class_to_idx": {
                "可回收物": 0,
                "有害垃圾": 1,
                ...
            }
        }
    """
    return jsonify({
        'categories': Config.CLASSES,
        'class_to_idx': Config.CLASS_TO_IDX
    })

@bp.route('/stats', methods=['GET'])
def get_stats():
    """
    获取分类统计数据
    
    请求方式：GET
    返回：
        {
            "total_predictions": 总预测次数,
            "category_distribution": {
                "可回收物": 次数,
                "有害垃圾": 次数,
                ...
            },
            "accuracy": 准确率
        }
    """
    # TODO: 实现统计功能
    stats = {
        'total_predictions': 0,
        'category_distribution': {
            category: 0 for category in Config.CLASSES
        },
        'accuracy': 0.0
    }
    return jsonify(stats) 