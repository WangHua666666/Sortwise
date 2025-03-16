import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from pathlib import Path
from ml.config import Config

# 加载环境变量配置
load_dotenv()

def create_app():
    """创建Flask应用实例"""
    # 创建Flask应用
    app = Flask(__name__)
    
    # 启用跨域资源共享
    CORS(app)
    
    # 创建上传目录
    upload_folder = Path(app.root_path) / 'static' / 'uploads'
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    # 配置应用
    app.config.update(
        # 密钥配置
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev'),
        # 上传文件配置
        UPLOAD_FOLDER=str(upload_folder),
        # 文件大小限制（16MB）
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        # 允许的文件类型
        ALLOWED_EXTENSIONS={'jpg', 'jpeg', 'png'},
        # API配置
        API_HOST=Config.API_CONFIG['host'],
        API_PORT=Config.API_CONFIG['port'],
        DEBUG=Config.API_CONFIG['debug']
    )
    
    # 注册主蓝图
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    # 注册API蓝图，设置URL前缀为/api
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    # 注册错误处理器
    from app.utils import error_handlers
    error_handlers.register_error_handlers(app)

    return app 