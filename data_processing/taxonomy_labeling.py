# -*- coding: utf-8 -*-
"""
taxonomy_labeling.py — 七大统计学求职方向标注
输入 : outputs/cleaned_jobs.csv
输出 : outputs/cleaned_jobs.csv  (新增 career_direction 列)
方法 : 基于 job_category / job_title / 关键词的优先级规则映射；
       与下游 SFT 的 7 大方向对齐，非统计相关岗位归入“其他(非目标)”。
"""
import re
import pandas as pd

PATH = "outputs/cleaned_jobs.csv"
df = pd.read_csv(PATH, encoding="utf-8-sig")

# 七大统计学方向的关键词规则（按优先级从上到下匹配）
RULES = [
    ("生物统计分析师", ["生物统计", "临床", "医药", "基因", "药物", "flimsy_placeholder"]),
    ("风控数据分析",   ["风控", "反欺诈", "信贷", "授信", "风险", "合规"]),
    ("BI报表工程师",   ["bi", "报表", "看板", "指标体系", "数据可视化", "tableau", "看数"]),
    ("商业分析BA",     ["商业分析", "business analy", "ba ", "商分", "策略分析", "经营分析"]),
    ("数据运营",       ["数据运营", "运营分析", "增长分析", "策略运营", "用户增长"]),
    ("算法建模岗",     ["算法", "机器学习", "深度学习", "建模", "推荐算法", "nlp",
                     "计算机视觉", "cv", "大模型", "模型", "策略算法", "预测"]),
    ("数据分析师",     ["数据分析", "数据科学", "data analy", "data scien",
                     "统计分析", "量化", "ab test", "a/b", "数据挖掘"]),
]

# 明确的非目标方向（工程/职能/设计等），避免误标
NON_TARGET = ["后端", "前端", "客户端", "测试", "运维", "硬件", "基础架构",
              "多媒体", "游戏客户端", "设计", "视觉设计", "ui", "人力", "hr",
              "财务", "法务", "行政", "销售", "采购", "供应链", "客服"]

def label(row):
    text = f"{row['job_category']} {row['job_title']}".lower()
    full = str(row.get("full_text", "")).lower()
    # 先看是否明显属于非目标方向（且标题里没有数据/分析/算法字样）
    if any(k in text for k in NON_TARGET) and not re.search(r"数据|分析|算法|风控|统计|bi|运营", text):
        return "其他(非目标)"
    # 按优先级规则匹配标题+类别
    for direction, kws in RULES:
        if any(k in text for k in kws):
            return direction
    # 标题未命中则退回到全文
    for direction, kws in RULES:
        if any(k in full for k in kws):
            return direction
    return "其他(非目标)"

df["career_direction"] = df.apply(label, axis=1)
df.to_csv(PATH, index=False, encoding="utf-8-sig")

print("======== 七大方向标注分布 ========")
print(df["career_direction"].value_counts().to_string())
target = df[df["career_direction"] != "其他(非目标)"]
print(f"\n统计学相关岗位(7方向)合计 : {len(target)} / {len(df)}  "
      f"({len(target)/len(df)*100:.1f}%)")

# 导出仅含七大方向的“核心数据集”，供下游人岗匹配 / SFT 使用
target.to_csv("outputs/cleaned_jobs_statistics.csv", index=False, encoding="utf-8-sig")
print("核心统计学数据集 -> outputs/cleaned_jobs_statistics.csv")
