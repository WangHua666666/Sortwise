# 导入必要的库
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum

# 创建数据库连接
# 使用环境变量获取数据库连接信息
DB_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)
# 创建数据库会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类
Base = declarative_base()

class WasteCategory(enum.Enum):
    """垃圾分类枚举
    根据数据集中的实际类别定义
    """
    RECYCLABLE = "可回收物"  # 如塑料瓶、玻璃瓶、纸张等
    HAZARDOUS = "有害垃圾"   # 如电池、药品、荧光灯等
    HOUSEHOLD = "其他垃圾"   # 如卫生纸、尿布等
    FOOD = "厨余垃圾"        # 如剩饭剩菜、果皮等

class Prediction(Base):
    """预测记录模型
    存储每次预测的结果，包括图像路径、类别ID、类别名称、置信度和创建时间
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)  # 主键
    image_path = Column(String(255), nullable=False)    # 图像路径
    category_id = Column(Integer, nullable=False)       # 类别ID
    category_name = Column(String(50), nullable=False)  # 类别名称
    subcategory = Column(String(50), nullable=False)    # 子类别（具体物品类型）
    confidence = Column(Float, nullable=False)          # 置信度
    created_at = Column(DateTime, default=datetime.utcnow)  # 创建时间

class Category(Base):
    """垃圾分类类别模型
    存储垃圾分类的类别信息，包括名称、描述和处理指南
    """
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)  # 主键
    name = Column(String(50), nullable=False)          # 类别名称
    main_category = Column(Enum(WasteCategory), nullable=False)  # 主类别
    description = Column(Text)                         # 类别描述
    disposal_guide = Column(Text)                      # 处理指南
    examples = Column(Text)                           # 示例物品

def get_db():
    """获取数据库会话
    用于在请求中获取数据库连接，并在请求结束后关闭连接
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 创建所有表
# 根据定义的模型创建数据库表
Base.metadata.create_all(bind=engine) 