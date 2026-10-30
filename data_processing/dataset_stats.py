# -*- coding: utf-8 -*-
"""
dataset_stats.py — 数据集统计与可视化
输出 : outputs/charts/*.png  (供报告与 PPT 使用)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

# 中文字体
fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

C = {"blue": "#2563eb", "teal": "#0d9488", "amber": "#d97706",
     "rose": "#e11d48", "slate": "#475569", "violet": "#7c3aed"}
PAL = ["#2563eb", "#0d9488", "#d97706", "#e11d48", "#7c3aed",
       "#0891b2", "#65a30d", "#db2777"]

df = pd.read_csv("outputs/cleaned_jobs.csv", encoding="utf-8-sig")
raw = pd.read_csv("data/all.csv", encoding="utf-8-sig")

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"outputs/charts/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# 1) 清洗漏斗
fig, ax = plt.subplots(figsize=(7, 3.6))
stages = ["原始爬取", "去重后", "质量过滤后\n(最终)"]
vals = [len(raw), 5128, len(df)]
bars = ax.barh(stages[::-1], vals[::-1], color=[C["teal"], C["blue"], C["slate"]][::-1])
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 60, b.get_y() + b.get_height()/2, f"{v:,}", va="center", fontsize=11)
ax.set_title("数据清洗漏斗：6788 → 5127（去重 1660 条，保留率 75.5%）", fontsize=12)
ax.set_xlim(0, len(raw)*1.12)
save(fig, "01_cleaning_funnel")

# 2) 平台分布
fig, ax = plt.subplots(figsize=(5, 4))
vc = df["company"].value_counts()
ax.pie(vc.values, labels=[f"{k}\n{v}条" for k, v in vc.items()],
       autopct="%1.1f%%", colors=PAL, startangle=90,
       wedgeprops=dict(width=0.45, edgecolor="w"))
ax.set_title("岗位来源平台分布（清洗后）", fontsize=12)
save(fig, "02_platform_pie")

# 3) 七大方向分布
fig, ax = plt.subplots(figsize=(7.5, 4))
order = ["数据分析师", "算法建模岗", "商业分析BA", "BI报表工程师",
         "风控数据分析", "数据运营", "生物统计分析师", "其他(非目标)"]
vc = df["career_direction"].value_counts().reindex(order).fillna(0)
bars = ax.bar(range(len(vc)), vc.values, color=PAL[:7] + ["#cbd5e1"])
ax.set_xticks(range(len(vc)))
ax.set_xticklabels(vc.index, rotation=30, ha="right", fontsize=9)
for i, v in enumerate(vc.values):
    ax.text(i, v + 20, int(v), ha="center", fontsize=9)
ax.set_title("七大统计学求职方向标注分布", fontsize=12)
ax.set_ylabel("岗位数")
save(fig, "03_taxonomy_dist")

# 4) Top 城市
fig, ax = plt.subplots(figsize=(7, 4))
vc = df["city_std"].value_counts().head(10)
bars = ax.barh(vc.index[::-1], vc.values[::-1], color=C["blue"])
for b, v in zip(bars, vc.values[::-1]):
    ax.text(v + 15, b.get_y()+b.get_height()/2, int(v), va="center", fontsize=9)
ax.set_title("Top 10 工作城市（标准化后）", fontsize=12)
save(fig, "04_top_cities")

# 5) Top 技能
sk = pd.read_csv("outputs/skills_dict.csv", encoding="utf-8-sig").head(15)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.barh(sk["skill"][::-1], sk["job_count"][::-1], color=C["violet"])
for b, v in zip(bars, sk["job_count"][::-1]):
    ax.text(v + 12, b.get_y()+b.get_height()/2, int(v), va="center", fontsize=8)
ax.set_title("Top 15 技能需求（覆盖岗位数）", fontsize=12)
save(fig, "05_top_skills")

# 6) 学历分布
fig, ax = plt.subplots(figsize=(5.5, 3.8))
vc = df["education_std"].value_counts()
ax.bar(range(len(vc)), vc.values, color=C["amber"])
ax.set_xticks(range(len(vc)))
ax.set_xticklabels(vc.index, rotation=20, ha="right", fontsize=9)
for i, v in enumerate(vc.values):
    ax.text(i, v+20, int(v), ha="center", fontsize=9)
ax.set_title("学历门槛分布", fontsize=12)
save(fig, "06_education")

# 7) JD 长度分布
fig, ax = plt.subplots(figsize=(6.5, 3.6))
lens = df["jd_text"].astype(str).str.len().clip(upper=1500)
ax.hist(lens, bins=40, color=C["teal"], edgecolor="white")
ax.axvline(lens.median(), color=C["rose"], ls="--", label=f"中位数={int(lens.median())}字")
ax.set_title("岗位描述(JD)文本长度分布", fontsize=12)
ax.set_xlabel("字符数"); ax.set_ylabel("岗位数"); ax.legend()
save(fig, "07_jd_length")

print("ALL CHARTS DONE")
