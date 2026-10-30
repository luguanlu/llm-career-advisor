# ranker.py
import os
import json
import pandas as pd
from user_profile import user_profile
from matching import calculate_rule_score
# 直接导入 retriever 中的 model 和余弦计算函数
from retriever import model, calculate_cosine_similarity

# =====================================================================
# 优化 1：【用户特征前置缓存】（避免在循环中重复编码用户，提速数十倍）
# =====================================================================
user_features = [
    user_profile['major'],
    user_profile['education'],
    " ".join(user_profile['skills']),
    user_profile['target_city'],
    user_profile['career_preference']
]
user_text = " ".join(user_features)
# 在最外层，对用户画像只进行 1 次编码，产生用户特征向量
user_vector = model.encode(user_text)

# 模拟数据（用于没有 jobs.csv 时的快速测试）
jobs_list = [
    {
        "job_id": "A160054",
        "job_title": "数据分析实习生（效果与创作）-抖音",
        "job_category": "产品 - 数据分析",
        "city": "北京",
        "requirements": "1、本科及以上学历在读，数学、统计学、经济学、计算机等相关专业优先； 2、逻辑思维和量化思维强，数据分析能力强，熟练使用Excel和SQL进行数据分析； 3、主动，有主人翁意识，执行力强，具备良好的内外部协作能力、表达能力。",
        "jd_text": "日常实习：面向全体在校生，为符合岗位要求的同学提供为期3个月及以上的项目实践机会。 团队介绍：抖音致力于打造一个开放、积极、多元、友善的平台... 1、参与抖音创作跟投稿相关方向的数据分析工作，搭建指标体系和数据看板，对业务表现进行敏捷有效的监控与评估...",
        "company": "字节跳动",
        "job_type": "实习"
    }
]
# 读取 A 爬取的数据（实际运行时请取消下面两行的注释，并注释掉 jobs_list 模拟数据）
df = pd.read_csv("jobs.csv")
jobs_list = df.to_dict(orient="records")

# =====================================================================
# 优化 2：【单层高开销循环】
# 在一轮循环中，一次性完成“规则计算”和“岗位向量提取”，并暂存中间体
# =====================================================================
temp_results = []
raw_similarities = []

for job in jobs_list:
    # 1. 计算基于业务规则的分数（计算开销低）
    rule_score, sub_scores = calculate_rule_score(user_profile, job)

    # 2. 净化岗位文本并提取向量（计算开销高，大模型编码）
    job_features = [
        job.get('job_title', ''),
        job.get('city', ''),
        job.get('requirements', ''),
        job.get('jd_text', '')
    ]
    job_text = " ".join(job_features)
    job_vector = model.encode(job_text)

    # 3. 计算与前置用户向量的余弦相似度
    raw_sim = calculate_cosine_similarity(user_vector, job_vector)
    raw_sim = max(0.0, float(raw_sim))

    raw_similarities.append(raw_sim)

    # 暂存轻量级中间数据，避免在后续归一化时重复运行高开销的 model.encode 和规则判断
    temp_results.append({
        "job": job,
        "rule_score": rule_score,
        "raw_sim": raw_sim
    })

# =====================================================================
# 优化 3：【极轻量后处理】
# 仅对暂存的分数进行纯数学的归一化拉伸和最终融合排序（执行时间接近 0 毫秒）
# =====================================================================
recommendations = []

if raw_similarities:
    min_sim = min(raw_similarities)
    max_sim = max(raw_similarities)
    sim_range = (max_sim - min_sim) if max_sim != min_sim else 1.0

    for item in temp_results:
        # 基于暂存的相似度，快速进行归一化拉伸
        semantic_score = ((item["raw_sim"] - min_sim) / sim_range) * 100
        semantic_score = round(float(semantic_score), 2)

        # 综合评分 (规则50% + 拉伸后语义分50%)
        final_score = round(0.5 * item["rule_score"] + 0.5 * semantic_score, 2)

        job = item["job"]
        rec_result = {
            "job_id": job["job_id"],
            "company": job["company"],
            "job_title": job["job_title"],
            "final_score": float(final_score),
            "rule_score": float(item["rule_score"]),
            "embedding_score": float(semantic_score),
            "skills_required": job.get("skills_required", []),
            "jd_text": job["jd_text"]
        }
        recommendations.append(rec_result)

# 排序并输出
recommendations = sorted(recommendations, key=lambda x: x["final_score"], reverse=True)

# 安全创建输出文件夹并写入结果
os.makedirs("outputs", exist_ok=True)
with open("outputs/topk_recommendations.json", "w", encoding="utf-8") as f:
    json.dump(recommendations[:5], f, ensure_ascii=False, indent=4)

print("推荐计算完成！")
if recommendations:
    print(f"Top 1 岗位: {recommendations[0]['job_title']}, 综合得分: {recommendations[0]['final_score']}")