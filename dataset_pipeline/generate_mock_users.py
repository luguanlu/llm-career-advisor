# generate_mock_users.py
import random
import os
import json

# 定义生成随机用户所需的元数据
MAJORS = ["统计学", "应用统计", "数理统计", "生物统计", "统计与数据科学",
          "数学", "应用数学", "计算数学", "基础数学", "数学与应用数学", "信息与计算科学",
          "计算机", "计算机科学", "计算机科学与技术", "软件工程", "数据科学", "数据科学与大数据技术", "人工智能", "大数据", "信息安全", "智能科学与技术",
          "经济学", "金融", "金融工程", "金融科技", "数量经济学", "计量经济学", "国际经济与贸易",
          "管理科学", "管理科学与工程", "信息管理与信息系统", "商业分析", "数据分析", "工业工程", "信息管理"]
EDUS = ["本科", "硕士", "博士"]
CITIES = ["北京", "上海", "深圳", "广州", "杭州", "成都"]
PREFERENCES = [
    "数据分析师",       # 数据分析方向 (对口基础岗)
    "算法建模岗",       # 机器学习/算法开发方向 (硕博高对口岗)
    "商业分析(BA)",     # 业务分析与商业咨询方向 (高薪复合岗)
    "BI报表工程师",     # 商业智能与数据可视化方向
    "风控数据分析师",   # 金融/风控量化建模方向
    "数据运营专员",     # 运营与用户行为分析方向
    "生物统计分析师"    # 医药、临床试验与基因工程方向 (生统对口岗)
]

def load_skill_pool():
    """
    动态从大模型生成的 tech_skills.json 中加载所有的真实技术栈。
    """
    dict_path = "tech_skills.json"

    # 兜底技能字典（防止文件读取失败或路径错误时系统报错崩溃）
    fallback_pool = ["SQL", "Python", "Excel", "R", "SPSS", "Tableau", "Hive", "Spark", "PPT", "PowerBI"]

    # 兼容在子目录运行的情形
    if not os.path.exists(dict_path):
        dict_path = os.path.join(os.path.dirname(__file__), "..", "tech_skills.json")

    if os.path.exists(dict_path):
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 提取出结构化数据中的每个 'skill' 字符串
                skills = [item["skill"] for item in data if "skill" in item]
                if skills:
                    # 进行去重，防止相同技能由于多次抽取导致模拟数据抽样重复
                    return list(set(skills))
        except Exception as e:
            print(f"警告：读取 tech_skills.json 失败（{e}），已启用备用兜底词典。")

    return fallback_pool

# 动态加载技能池
SKILL_POOL = load_skill_pool()


def generate_one_random_user():
    """随机生成一个求职者画像"""

    interested_cats = random.sample(PREFERENCES, 2)

    return {
        "major": random.choice(MAJORS),
        "education": random.choice(EDUS),
        "skills": random.sample(SKILL_POOL, random.randint(2, 5)),
        "target_city": random.choice(CITIES),
        "interested_categories": interested_cats, # 随机选两个意向
        "career_preference": random.choice(interested_cats)
    }

if __name__ == "__main__":
    # 测试生成 3 个看看
    for i in range(3):
        print(generate_one_random_user())