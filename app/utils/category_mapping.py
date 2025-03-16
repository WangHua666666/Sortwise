# 类别映射
CATEGORY_MAPPING = {
    # 可回收物
    'plasticbottle': {'name': '塑料瓶', 'main_category': '可回收物'},
    'glassbottle': {'name': '玻璃瓶', 'main_category': '可回收物'},
    'newspaper': {'name': '报纸', 'main_category': '可回收物'},
    'carton': {'name': '纸箱', 'main_category': '可回收物'},
    'cans': {'name': '易拉罐', 'main_category': '可回收物'},
    'milkbox': {'name': '牛奶盒', 'main_category': '可回收物'},
    'bowlsanddishes': {'name': '碗碟', 'main_category': '可回收物'},
    
    # 有害垃圾
    'battery': {'name': '电池', 'main_category': '有害垃圾'},
    'bulb': {'name': '灯泡', 'main_category': '有害垃圾'},
    'thermometer': {'name': '温度计', 'main_category': '有害垃圾'},
    'medicinebottle': {'name': '药瓶', 'main_category': '有害垃圾'},
    'nailpolishbottle': {'name': '指甲油瓶', 'main_category': '有害垃圾'},
    'pesticidebottle': {'name': '农药瓶', 'main_category': '有害垃圾'},
    'traditionalChinesemedicine': {'name': '中药', 'main_category': '有害垃圾'},
    'tabletcapsule': {'name': '药片胶囊', 'main_category': '有害垃圾'},
    'XLight': {'name': '荧光灯', 'main_category': '有害垃圾'},
    
    # 其他垃圾
    'bandaid': {'name': '创可贴', 'main_category': '其他垃圾'},
    'toothbrush': {'name': '牙刷', 'main_category': '其他垃圾'},
    'toothpastetube': {'name': '牙膏管', 'main_category': '其他垃圾'},
    'diapers': {'name': '尿布', 'main_category': '其他垃圾'},
    'napkin': {'name': '餐巾纸', 'main_category': '其他垃圾'},
    'facialmask': {'name': '口罩', 'main_category': '其他垃圾'},
    'cigarettebutt': {'name': '烟头', 'main_category': '其他垃圾'},
    'plasticene': {'name': '橡皮泥', 'main_category': '其他垃圾'},
    'toothpick': {'name': '牙签', 'main_category': '其他垃圾'},
    'rag': {'name': '抹布', 'main_category': '其他垃圾'},
    
    # 厨余垃圾
    'leftovers': {'name': '剩饭剩菜', 'main_category': '厨余垃圾'},
    'watermelonrind': {'name': '西瓜皮', 'main_category': '厨余垃圾'},
    'bread': {'name': '面包', 'main_category': '厨余垃圾'},
}

# 主类别描述
MAIN_CATEGORY_INFO = {
    '可回收物': {
        'description': '可以回收利用的垃圾，包括各种废纸、塑料、玻璃、金属和布料等',
        'disposal_guide': '清洁干燥后投放到可回收物收集容器中',
        'examples': '报纸、纸箱、塑料瓶、玻璃瓶、易拉罐、旧衣物等'
    },
    '有害垃圾': {
        'description': '对人体健康或者自然环境造成直接或潜在危害的废弃物',
        'disposal_guide': '投放到有害垃圾收集容器中，不要与其他垃圾混合',
        'examples': '废电池、废荧光灯管、废药品、废油漆、废杀虫剂等'
    },
    '其他垃圾': {
        'description': '除可回收物、有害垃圾、厨余垃圾以外的其他生活废弃物',
        'disposal_guide': '投放到其他垃圾收集容器中',
        'examples': '卫生纸、尿布、烟头、陶瓷碎片等'
    },
    '厨余垃圾': {
        'description': '日常生活中产生的食物残余和食品加工废料',
        'disposal_guide': '沥干水分后投放到厨余垃圾收集容器中',
        'examples': '剩菜剩饭、果皮、蛋壳、茶叶渣等'
    }
} 