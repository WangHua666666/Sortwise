# 导入应用工厂函数
from app import create_app
from ml.config import Config

def main():
    """
    启动Flask应用程序
    使用配置类中定义的设置
    """
    app = create_app()
    app.run(
        host=Config.API_CONFIG['host'],
        port=Config.API_CONFIG['port'],
        debug=Config.API_CONFIG['debug']
    )

if __name__ == '__main__':
    main() 