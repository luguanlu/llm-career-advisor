import json
import os
import pandas as pd
from generate_mock_users import generate_one_random_user
from matching import calculate_rule_score
from retriever import model, calculate_cosine_similarity
import torch
print(f"✅ 当前可见的物理显卡数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"✅ 正在使用的物理显卡名称: {torch.cuda.get_device_name(0)}")
import openai
import time # 引入 time 模块用于防抖重试

# =====================================================================
# 1. 读取 CSV 岗位库并预先计算所有岗位向量（极大提升性能！）
# =====================================================================
df = pd.read_csv("all.csv")
jobs_list = df.to_dict(orient="records")

print("正在预计算所有岗位特征向量，请稍候...")
# 提前计算并缓存所有岗位的向量，避免在内层循环中重复计算
for job in jobs_list:
    job_features = [str(job.get('job_title', '')), str(job.get('city', '')), str(job.get('requirements', '')), str(job.get('jd_text', ''))]
    job_text = " ".join(job_features)
    job['vector'] = model.encode(job_text)
print("岗位特征向量预计算完成！\n")


# =====================================================================
# 2. 算法升级：为用户计算并筛选最匹配的 Top-K 个岗位 (K=3)
# =====================================================================
def get_top_k_jobs_for_user(user, jobs, k=3):
    user_features = [user['major'], user['education'], " ".join(user['skills']), user['target_city'], user['career_preference']]
    user_text = " ".join(user_features)
    user_vector = model.encode(user_text)

    scored_jobs = []

    for job in jobs:
        rule_score, _ = calculate_rule_score(user, job)

        # 直接使用预先计算好的向量！
        job_vector = job['vector']
        raw_sim = calculate_cosine_similarity(user_vector, job_vector)
        raw_sim = max(0.0, float(raw_sim))

        total_score = 0.5 * rule_score + 0.5 * (raw_sim * 100)
        scored_jobs.append((total_score, job))

    scored_jobs = sorted(scored_jobs, key=lambda x: x[0], reverse=True)
    top_k_jobs = [item[1] for item in scored_jobs[:k]]

    return top_k_jobs


# =====================================================================
# 3. 核心生成逻辑：生成 300 位用户
# =====================================================================
sft_inputs = []
NUM_USERS = 300

print(f"正在生成 {NUM_USERS} 名求职者，并使用 B 的推荐算法为每人匹配 Top-3 岗位...")
for i in range(NUM_USERS):
    user = generate_one_random_user()
    top_3_jobs = get_top_k_jobs_for_user(user, jobs_list, k=3)

    for rank, job in enumerate(top_3_jobs, 1):
        input_text = (
            f"【求职者背景】专业：{user['major']}；学历：{user['education']}；掌握技能：{', '.join(user['skills'])}；意向城市：{user['target_city']}；职业偏好：{user['career_preference']}。\n"
            f"【推荐岗位（第 {rank} 匹配）】公司：{job['company']}；岗位：{job['job_title']}；城市：{job['city']}；岗位要求技能：Excel、SQL、A/B Test、数据清洗等；"
            f"岗位职责简述：{job['jd_text'][:500] if len(job['jd_text']) > 500 else job['jd_text']}"
        )

        sft_inputs.append({
            "instruction": "你是一个专业的统计学求职导师。请根据求职者的个人背景和推荐岗位信息，使用中文生成推荐理由、技能缺口分析和背景提升建议。请务必以标准的 JSON 格式输出（注意：JSON 的键名保持英文如 reason, gap, suggestion，但键值 Value 必须全部使用中文进行详细描述，不要包含任何寒暄和口水话）。",
            "input": input_text
        })

    if (i+1) % 50 == 0:
        print(f"已匹配并打包 {i+1} 名用户（共计 {len(sft_inputs)} 条对偶数据）...")

print(f"\n基础数据打包完毕，共计：{len(sft_inputs)} 条。")


# =====================================================================
# 4. 调用大模型进行知识蒸馏（生成 output）
# =====================================================================
api_key = os.getenv("LLM_API_KEY")
if not api_key:
    raise RuntimeError("Set LLM_API_KEY before running this script.")

client = openai.OpenAI(
    api_key=api_key,
    base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
)

final_sft_dataset = []

print("\n开始调用先进大模型生成中文专家解释（知识蒸馏）...")
for idx, item in enumerate(sft_inputs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": "你是一个严格按照 JSON 格式输出的职业生涯规划专家。请务必使用中文（Chinese）来撰写 JSON 里的所有具体内容（Value），严禁在 JSON 字段值中使用英文进行长句解释。"},
                    {"role": "user", "content": f"{item['instruction']}\n\n输入数据：{item['input']}"}
                ],
                temperature=0.3,
                timeout=15 # 设置超时时间，防止卡死
            )

            llm_output = response.choices[0].message.content.strip()

            final_sft_dataset.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": llm_output
            })

            if (idx + 1) % 10 == 0:
                print(f"已蒸馏完成 {idx + 1}/{len(sft_inputs)} 条专家数据...")
            break # 成功则跳出重试循环

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"第 {idx} 条数据调用超时或失败，正在重试 ({attempt+1}/{max_retries})...")
                time.sleep(2) # 暂停2秒再重试
            else:
                print(f"第 {idx} 条数据最终调用失败: {e}")

# 保存最终数据，过滤掉因为 API 失败导致没有输出的字典键
final_sft_dataset = [d for d in final_sft_dataset if 'output' in d]

with open("tech_advisor_ooo_data.json", "w", encoding="utf-8") as f:
    json.dump(final_sft_dataset, f, ensure_ascii=False, indent=4)

print(f"\n🎉 恭喜！高精度的 {len(final_sft_dataset)} 条中文微调数据集已成功写入 tech_advisor_ooo_data.json 文件！")
