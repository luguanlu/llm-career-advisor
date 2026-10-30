# -*- coding: utf-8 -*-
"""
skill_extraction.py — 技能词典构建 (skills_dict / tech_skills.json)
输入 : outputs/cleaned_jobs.csv
输出 : outputs/skills_dict.csv, outputs/tech_skills.json
方法 : 种子词典(正则边界抽取) + JD 词频统计 + 人工校验类别归并；
       并为每个岗位打上命中的技能标签，供下游人岗匹配的硬技能对齐。
"""
import re
import json
import pandas as pd
from collections import Counter

PATH = "outputs/cleaned_jobs.csv"
df = pd.read_csv(PATH, encoding="utf-8-sig")

# ----------------------------------------------------------------------
# 1. 人工构建的种子技能库（按类别归并，经高频词统计校验后确定）
#    key = 标准技能名, value = 匹配用的别名/正则片段
# ----------------------------------------------------------------------
SKILL_LEXICON = {
    "编程语言": {
        "Python": r"python", "R": r"\bR\b|R语言", "SQL": r"sql",
        "Java": r"\bjava\b", "C++": r"c\+\+", "Scala": r"scala", "Shell": r"shell|bash",
    },
    "数据处理与大数据": {
        "Hive": r"hive", "Spark": r"spark", "Hadoop": r"hadoop",
        "Flink": r"flink", "Pandas": r"pandas", "MapReduce": r"mapreduce",
        "ETL": r"\betl\b", "Kafka": r"kafka",
    },
    "机器学习与算法": {
        "机器学习": r"机器学习|machine learning", "深度学习": r"深度学习|deep learning",
        "PyTorch": r"pytorch", "TensorFlow": r"tensorflow",
        "NLP": r"nlp|自然语言", "计算机视觉": r"计算机视觉|cv\b|图像",
        "推荐系统": r"推荐系统|推荐算法", "大模型": r"大模型|llm|大语言模型",
        "特征工程": r"特征工程",
    },
    "统计与实验": {
        "统计分析": r"统计分析|统计学", "A/B Test": r"a/?b\s?test|abtest|ab实验|a/b实验",
        "假设检验": r"假设检验|显著性", "因果推断": r"因果推断|causal",
        "回归分析": r"回归", "时间序列": r"时间序列",
    },
    "BI与可视化": {
        "Tableau": r"tableau", "PowerBI": r"power\s?bi",
        "Excel": r"excel", "看板/报表": r"看板|报表|dashboard",
        "指标体系": r"指标体系",
    },
    "业务与领域": {
        "数据挖掘": r"数据挖掘|data mining", "用户画像": r"用户画像",
        "风控建模": r"风控|反欺诈", "数据仓库": r"数据仓库|数仓",
    },
}

# 扁平化为 (标准名, 类别, 正则)
flat = []
for cat, d in SKILL_LEXICON.items():
    for name, pat in d.items():
        flat.append((name, cat, re.compile(pat, re.IGNORECASE)))

# ----------------------------------------------------------------------
# 2. 统计每个技能在语料中的出现岗位数（document frequency）
# ----------------------------------------------------------------------
texts = df["full_text"].fillna("").astype(str).tolist()
freq = Counter()
tagged = []
for t in texts:
    hits = set()
    for name, cat, pat in flat:
        if pat.search(t):
            hits.add(name)
            freq[name] += 1
    tagged.append("|".join(sorted(hits)))

df["skill_tags"] = tagged
df.to_csv(PATH, index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------
# 3. 导出技能词典 CSV（含类别 + 频次 + 覆盖率）
# ----------------------------------------------------------------------
rows = []
name2cat = {name: cat for name, cat, _ in flat}
for name in name2cat:
    rows.append({
        "skill": name,
        "category": name2cat[name],
        "job_count": freq.get(name, 0),
        "coverage_%": round(freq.get(name, 0) / len(df) * 100, 1),
    })
dict_df = pd.DataFrame(rows).sort_values("job_count", ascending=False)
dict_df.to_csv("outputs/skills_dict.csv", index=False, encoding="utf-8-sig")

# tech_skills.json：按类别组织，供下游匹配算法读取
tech_json = {cat: sorted([n for n in d]) for cat, d in SKILL_LEXICON.items()}
with open("outputs/tech_skills.json", "w", encoding="utf-8") as f:
    json.dump(tech_json, f, ensure_ascii=False, indent=2)

print("======== 技能词典 Top 20（按覆盖岗位数）========")
print(dict_df.head(20).to_string(index=False))
print(f"\n技能总数 : {len(dict_df)}  |  类别数 : {len(SKILL_LEXICON)}")
print(f"平均每岗位命中技能数 : {df['skill_tags'].map(lambda s: len(s.split('|')) if s else 0).mean():.2f}")
print("输出 -> outputs/skills_dict.csv, outputs/tech_skills.json")
