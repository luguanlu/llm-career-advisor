# -*- coding: utf-8 -*-
"""
data_cleaning.py  — 岗位数据清洗与字段标准化
输入 : data/all.csv        (三平台爬虫合并后的原始数据)
输出 : outputs/cleaned_jobs.csv
步骤 : 1) 结构统一   2) 文本清洗   3) 去重
       4) 城市/学历/专业/经验/时间 标准化   5) 质量过滤
"""
import re
import pandas as pd
import numpy as np

RAW_PATH = "data/all.csv"
OUT_PATH = "outputs/cleaned_jobs.csv"

# ----------------------------------------------------------------------
# 0. 读取
# ----------------------------------------------------------------------
df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
n_raw = len(df)
report = {"原始条数": n_raw}

# ----------------------------------------------------------------------
# 1. 修复 job_id（腾讯被 pandas 读成科学计数法；美团缺失 -> 用 URL 兜底）
# ----------------------------------------------------------------------
def fix_job_id(row):
    jid = row.get("job_id")
    url = str(row.get("detail_url", ""))
    # 腾讯：postId 在 URL 里，最准确
    m = re.search(r"postId=(\d+)", url)
    if m:
        return "TX_" + m.group(1)
    # 字节：position/<id>/detail
    m = re.search(r"position/(\d+)/detail", url)
    if m:
        return "BD_" + m.group(1)
    # 美团：jobUnionId=<id>
    m = re.search(r"jobUnionId=(\d+)", url)
    if m:
        return "MT_" + m.group(1)
    # 兜底：原 job_id
    if pd.notna(jid):
        return str(jid)
    return np.nan

df["job_id"] = df.apply(fix_job_id, axis=1)

# ----------------------------------------------------------------------
# 2. 文本清洗：去 HTML / 多余空白 / 控制字符
# ----------------------------------------------------------------------
def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t)
    t = re.sub(r"<[^>]+>", " ", t)                 # HTML 标签
    t = re.sub(r"&[a-zA-Z]+;", " ", t)             # HTML 实体
    t = t.replace("\u3000", " ").replace("\xa0", " ")
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

for col in ["job_title", "requirements", "jd_text", "job_category"]:
    df[col] = df[col].map(clean_text)

# 合并 requirements + jd_text 为全文，供技能抽取/标注使用
df["full_text"] = (df["job_title"] + " " + df["job_category"] + " "
                   + df["requirements"] + " " + df["jd_text"]).str.strip()

# ----------------------------------------------------------------------
# 3. 去重：先去完全重复的 job_id，再按 (标题+公司+JD) 语义去重
# ----------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=["job_id"], keep="first")
report["按job_id去重后"] = len(df)
df = df.drop_duplicates(subset=["job_title", "company", "jd_text"], keep="first")
report["按(标题+公司+JD)去重后"] = len(df)
report["累计删除重复"] = before - len(df)

# ----------------------------------------------------------------------
# 4. 字段标准化
# ----------------------------------------------------------------------
# 4.1 城市：去“市”后缀、拆多城市取首个、异常置“未知”
def norm_city(c):
    c = clean_text(c)
    if not c or c in ("未知", "nan"):
        return "未知"
    c = re.split(r"[、,，/|]", c)[0]        # 多城市取第一个
    c = c.replace("市", "").strip()
    alias = {"深圳市": "深圳", "上海市": "上海", "北京市": "北京"}
    return alias.get(c, c) if c else "未知"

df["city_std"] = df["city"].map(norm_city)

# 4.2 学历：从 requirements 抽取最低学历要求
EDU_ORDER = ["大专", "本科", "硕士", "博士"]
def extract_edu(text):
    text = str(text)
    hit = []
    if re.search(r"博士", text): hit.append("博士")
    if re.search(r"硕士|研究生", text): hit.append("硕士")
    if re.search(r"本科", text): hit.append("本科")
    if re.search(r"大专|专科", text): hit.append("大专")
    if not hit:
        return "不限/未标注"
    # 招聘中“本科及以上/硕士及以上”通常写最低门槛，取最低者
    return sorted(hit, key=lambda x: EDU_ORDER.index(x))[0]

df["education_std"] = df["requirements"].map(extract_edu)

# 4.3 专业：MAJOR_SYNONYMS 归并为 5 大学科群
MAJOR_SYNONYMS = {
    "统计学": ["统计", "应用统计", "生物统计", "统计学"],
    "数学": ["数学", "应用数学", "运筹", "概率"],
    "计算机": ["计算机", "软件", "人工智能", "自动化", "电子工程", "通信"],
    "经济金融": ["经济", "金融", "财经", "计量"],
    "管理与商业分析": ["管理", "商业分析", "工商", "市场营销"],
}
def extract_majors(text):
    text = str(text)
    got = [m for m, kws in MAJOR_SYNONYMS.items() if any(k in text for k in kws)]
    return "|".join(got) if got else "未标注"

df["major_std"] = df["requirements"].map(extract_majors)

# 4.4 工作经验：统一为“实习/应届/1年以上/不限”
def norm_exp(row):
    e = clean_text(row.get("work_experience"))
    jt = clean_text(row.get("job_type"))
    text = (e + " " + jt).strip()
    if not text or text == "nan":
        return "实习/应届"          # 字节校招页默认实习
    if "不限" in text:
        return "不限"
    m = re.search(r"(\d+)\s*年", text)
    if m:
        return f"{m.group(1)}年以上"
    if "实习" in text:
        return "实习/应届"
    return "实习/应届"

df["experience_std"] = df.apply(norm_exp, axis=1)

# 4.5 发布时间：统一为 YYYY-MM-DD（缺失保留空）
def norm_date(d):
    d = str(d)
    m = re.search(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})", d)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return ""

df["publish_date_std"] = df["publish_time"].map(norm_date)

# ----------------------------------------------------------------------
# 5. 质量过滤：JD 过短（<30字）或标题为空 -> 视为无效
# ----------------------------------------------------------------------
before = len(df)
df = df[df["job_title"].str.len() > 0]
df = df[df["jd_text"].str.len() >= 30]
report["质量过滤删除"] = before - len(df)
report["最终有效条数"] = len(df)

# ----------------------------------------------------------------------
# 6. 导出（保留标准化字段 + 原始关键字段）
# ----------------------------------------------------------------------
keep = ["job_id", "company", "job_title", "job_category",
        "city_std", "education_std", "major_std", "experience_std",
        "publish_date_std", "search_keyword",
        "requirements", "jd_text", "full_text", "detail_url"]
df_out = df[keep].reset_index(drop=True)
df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------
# 7. 打印清洗审计报告
# ----------------------------------------------------------------------
print("======== 清洗审计报告 ========")
for k, v in report.items():
    print(f"{k:>18} : {v}")
print(f"{'保留率':>18} : {len(df)/n_raw*100:.1f}%")
print(f"输出 -> {OUT_PATH}")
