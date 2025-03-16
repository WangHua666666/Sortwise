import os
from flask import Blueprint, request, jsonify, render_template, current_app
from werkzeug.utils import secure_filename
from ml.models.inference import Predictor

# 创建蓝图
bp = Blueprint('main', __name__)

# 创建预测器实例
predictor = Predictor()

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@bp.route('/')
def index():
    """主页"""
    return render_template('index.html')

@bp.route('/classify', methods=['POST'])
def classify():
    """处理图片分类请求"""
    # 检查是否有文件上传
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '没有上传文件'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    # 检查文件类型并处理
    if file and allowed_file(file.filename):
        try:
            # 保存上传的文件
            filename = secure_filename(file.filename)
            upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # 进行预测
            result = predictor.predict(filepath)
            
            # 添加图片URL到结果中
            if result['success']:
                result['image_url'] = f'/static/uploads/{filename}'
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': '不支持的文件类型'})

@bp.route('/about')
def about():
    """渲染关于页面"""
    return render_template('about.html')

if __name__ == '__main__':
    bp.run(debug=True) 