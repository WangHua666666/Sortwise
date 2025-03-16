import os
import sys
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# 加载环境变量
load_dotenv()

def init_database():
    """
    初始化数据库
    """
    try:
        # 连接MySQL服务器
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )

        if connection.is_connected():
            cursor = connection.cursor()
            
            # 创建数据库
            db_name = os.getenv('DB_NAME')
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            cursor.execute(f"USE {db_name}")

            # 创建预测记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    image_path VARCHAR(255) NOT NULL,
                    category_id INT NOT NULL,
                    category_name VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建类别表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(50) NOT NULL,
                    description TEXT,
                    disposal_guide TEXT
                )
            """)

            # 插入默认类别
            categories = [
                (0, '可回收物', '可以回收利用的垃圾', '清洁干燥，分类投放到对应的回收箱'),
                (1, '有害垃圾', '对人体健康或自然环境造成直接或潜在危害的垃圾', '投放到有害垃圾收集点，不要与其他垃圾混合'),
                (2, '厨余垃圾', '日常生活中产生的食物残余', '沥干水分，投放到厨余垃圾收集桶'),
                (3, '其他垃圾', '除可回收物、有害垃圾、厨余垃圾以外的其他生活垃圾', '投放到其他垃圾收集桶')
            ]

            cursor.execute("TRUNCATE TABLE categories")
            cursor.executemany("""
                INSERT INTO categories (id, name, description, disposal_guide)
                VALUES (%s, %s, %s, %s)
            """, categories)

            connection.commit()
            print("数据库初始化成功！")

    except Error as e:
        print(f"数据库初始化失败: {e}")
        sys.exit(1)

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == '__main__':
    init_database() 