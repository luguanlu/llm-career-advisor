# src/matching.py
import os
import json
import re

# =====================================================================
# 1. 扩展后的专业同义词/关联树词典 (MAJOR_SYNONYMS)
# =====================================================================
MAJOR_SYNONYMS = {
    "统计学": [
        "统计学", "统计", "应用统计", "数理统计", "生物统计", "统计与数据科学",
        "应用统计学", "biostatistics", "statistics", "stats", "数据统计"
    ],
    "数学": [
        "数学", "应用数学", "计算数学", "基础数学", "数学与应用数学", "信息与计算科学",
        "math", "mathematics"
    ],
    "计算机": [
        "计算机", "计算机科学", "计算机科学与技术", "软件工程", "数据科学",
        "数据科学与大数据技术", "人工智能", "ai", "computer science", "cs",
        "software engineering", "大数据", "信息安全", "智能科学与技术"
    ],
    "经济金融": [
        "经济", "经济学", "金融", "金融工程", "金融科技", "数量经济学",
        "计量经济学", "国际经济与贸易", "economics", "finance", "量化投资"
    ],
    "管理与商业分析": [
        "管理科学", "管理科学与工程", "信息管理与信息系统", "商业分析",
        "business analytics", "ba", "数据分析", "工业工程", "信息管理"
    ]
}

# =====================================================================
# 2. 动态加载大模型生成的 tech_skills.json
# =====================================================================
def load_tech_skills():
    """
    读取大模型生成的结构化 tech_skills.json 文件并提取出扁平化的技能列表。
    格式要求：[ {"category": "xxx", "skill": "Python"}, ... ]
    """
    dict_path = "tech_skills.json"
    fallback_skills = ["SQL", "Excel", "Python", "R", "SPSS", "Tableau", "Hive", "Spark"]

    if not os.path.exists(dict_path):
        # 兼容脚本在 src 目录下执行的情况
        dict_path = os.path.join(os.path.dirname(__file__), "..", "tech_skills.json")

    if os.path.exists(dict_path):
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 从结构化字典列表中，提取出所有的 'skill' 字符串
                skills_list = [item["skill"] for item in data if "skill" in item]
                if skills_list:
                    return skills_list
        except Exception as e:
            print(f"警告：读取 tech_skills.json 失败（{e}），将使用默认技能列表。")

    return fallback_skills


def is_skill_in_text(skill, text):
    """
    【高级防误判机制】：
    避免像 "R"、"C" 这样的单字母/短字母技能，在匹配 "Requirements"、"Company" 时发生误判。
    """
    if len(skill) <= 2:
        # 使用正则表达式匹配独立的单词边界（如：\bR\b）
        pattern = rf"(?i)\b{re.escape(skill)}\b"
        return bool(re.search(pattern, text))
    else:
        # 长于 2 个字符的技能，直接使用不区分大小写的子串匹配
        return skill.lower() in text.lower()


# =====================================================================
# 3. 专业匹配判断逻辑 (多对多模糊匹配)
# =====================================================================
def check_major_match(user_major, job_requirements_text):
    """
    判断用户的专业大类是否与招聘要求的专业匹配。
    """
    # 1. 识别用户所属的专业大类
    matched_user_categories = []
    for core_major, synonyms in MAJOR_SYNONYMS.items():
        if any(syn.lower() in user_major.lower() for syn in synonyms):
            matched_user_categories.append(core_major)

    # 如果用户的专业太冷门，没在我们的大类里，默认给 0 分（或者视情况给个基础分）
    if not matched_user_categories:
        return False

    # 2. 检查招聘要求文本中，是否提到了用户所属专业大类中的任意一个同义词
    for category in matched_user_categories:
        synonyms = MAJOR_SYNONYMS[category]
        if any(syn.lower() in job_requirements_text.lower() for syn in synonyms):
            return True

    return False


# =====================================================================
# 4. 主打分函数 (calculate_rule_score)
# =====================================================================
def calculate_rule_score(user, job):
    """
    计算基于业务规则的人岗匹配总分（权重之和为 100%）。
    """
    job_requirements_text = str(job.get("requirements", ""))

    # ---------------- 5.1 技能匹配度 (权重 35%) ----------------
    skills_list = load_tech_skills()

    # 筛选出当前岗位 JD 中提到了哪些词典里的技能
    job_skills = [s for s in skills_list if is_skill_in_text(s, job_requirements_text)]

    if len(job_skills) == 0:
        skill_score = 100  # 如果岗位没有提任何硬技能，默认不在这项扣分
    else:
        matched_skills = set(user["skills"]).intersection(set(job_skills))
        skill_score = (len(matched_skills) / len(job_skills)) * 100

    # ---------------- 5.2 专业匹配度 (权重 20%) ----------------
    is_major_matched = check_major_match(user["major"], job_requirements_text)
    major_score = 100 if is_major_matched else 0

    # ---------------- 5.3 城市匹配度 (权重 15%) ----------------
    city_score = 100 if user["target_city"] == job.get("city", "") else 0

    # ---------------- 5.4 意向方向匹配度 (权重 15%) ----------------
    interest_score = 0
    for category in user["interested_categories"]:
        if category in str(job.get("job_title", "")) or category in str(job.get("job_category", "")):
            interest_score = 100
            break

    # ---------------- 5.5 学历错配与强惩罚机制 (权重 15%) ----------------
    edu_map = {"本科": 1, "硕士": 2, "博士": 3}
    user_edu_level = edu_map.get(user["education"], 0)

    # 提取岗位最低学历要求
    job_edu_req = "本科"  # 默认最低要求
    if "硕士" in job_requirements_text:
        job_edu_req = "硕士"
    elif "博士" in job_requirements_text:
        job_edu_req = "博士"
    job_edu_level = edu_map.get(job_edu_req, 1)

    # 学历差值 = 用户学历 - 岗位学历
    edu_diff = user_edu_level - job_edu_level

    if edu_diff == 0:
        # A. 完美匹配 (硕士投硕士岗)
        education_score = 100
    elif edu_diff > 0:
        # B. 高配低就折价 (避免过度匹配)
        if edu_diff == 1:
            education_score = 50   # 硕士投本科岗，打折给 50 分
        else: # edu_diff == 2
            education_score = 10   # 博士投本科岗，大材小用，仅给 10 分
    else:
        # C. 未达标强力惩罚 (硬性门槛过滤)
        if edu_diff == -1:
            education_score = -200  # 本科投硕士岗，扣大分 (0.15 * -200 = 实际总分扣 30 分)
        else: # edu_diff == -2
            education_score = -500  # 本科投博士岗，一票否决 (0.15 * -500 = 实际总分扣 75 分)

    # ---------------- 6. 最终加权汇总 ----------------
    final_rule_score = (
        0.35 * skill_score +
        0.20 * major_score +
        0.15 * city_score +
        0.15 * interest_score +
        0.15 * education_score
    )

    # 为保证数据干净，所有子分数取原生 float
    return round(float(final_rule_score), 2), {
        "skill_score": round(float(skill_score), 2),
        "major_score": float(major_score),
        "city_score": float(city_score),
        "interest_score": float(interest_score),
        "education_score": float(education_score)
    }