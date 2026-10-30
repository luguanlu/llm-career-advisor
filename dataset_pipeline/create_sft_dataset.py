# create_sft_dataset.py
import json
import os
import pandas as pd
from generate_mock_users import generate_one_random_user
from matching import calculate_rule_score
# 导入我们之前写好的、性能极高的向量计算模型
from retriever import model, calculate_cosine_similarity

# 读取 A 爬取好的岗位数据
df = pd.read_csv("jobs.csv")
jobs_list = df.to_dict(orient="records")

def get_best_job_for_user(user, jobs):
    """
    运行成员 B 的算法，为该用户选出最匹配的 Top-1 岗位
    """
    # 1. 缓存用户向量
    user_features = [user['major'], user['education'], " ".join(user['skills']), user['target_city'], user['career_preference']]
    user_text = " ".join(user_features)
    user_vector = model.encode(user_text)

    best_job = None
    best_score = -9999

    for job in jobs:
        # 计算规则分
        rule_score, _ = calculate_rule_score(user, job)

        # 计算语义分
        job_features = [job.get('job_title', ''), job.get('city', ''), job.get('requirements', ''), job.get('jd_text', '')]
        job_text = " ".join(job_features)
        job_vector = model.encode(job_text)
        raw_sim = calculate_cosine_similarity(user_vector, job_vector)
        raw_sim = max(0.0, float(raw_sim))

        # 简单模拟融合（这里不需要全局 Min-Max，简单相加即可找出最匹配的）
        total_score = 0.5 * rule_score + 0.5 * (raw_sim * 100)

        if total_score > best_score:
            best_score = total_score
            best_job = job

    return best_job

# =====================================================================
# 核心生成循环（生成 500 条 input 数据）
# =====================================================================
sft_inputs = []

print("正在生成 500 条高匹配度的人岗对偶数据...")
for i in range(5):
    user = generate_one_random_user()
    best_job = get_best_job_for_user(user, jobs_list)

    # 打包成大模型看得懂的 input 文本
    input_text = (
        f"【求职者背景】专业：{user['major']}；学历：{user['education']}；掌握技能：{', '.join(user['skills'])}；意向城市：{user['target_city']}；职业偏好：{user['career_preference']}。\n"
        f"【推荐岗位】公司：{best_job['company']}；岗位：{best_job['job_title']}；城市：{best_job['city']}；学历要求：本科；岗位要求技能：Excel、SQL、A/B Test、指标体系等；"
        f"岗位职责：{best_job['jd_text'][:100]}..." # 截取前 100 字，防止文本过长
    )

    sft_inputs.append({
        "instruction": "你是一个专业的统计学求职导师。请根据求职者的个人背景和推荐岗位信息，使用中文生成推荐理由、技能缺口分析和背景提升建议。请务必以标准的 JSON 格式输出（注意：JSON 的键名保持英文如 reason, gap, suggestion，但键值 Value 必须全部使用中文进行详细描述，不要包含任何寒暄和口水话）。",
        "input": input_text,
        "raw_user": user,       # 暂存，方便后续使用
        "raw_job": best_job     # 暂存，方便后续使用
    })

    if (i+1) % 50 == 0:
        print(f"已生成 {i+1}/500 条基础数据...")

import openai # 或者使用对应大模型厂家的 SDK

# 配置你的大模型 API 密钥（以智谱 AI / 零一万物 / DeepSeek 为例）
api_key = os.getenv("LLM_API_KEY")
if not api_key:
    raise RuntimeError("Set LLM_API_KEY before running this script.")

client = openai.OpenAI(
    api_key=api_key,
    base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
)

final_sft_dataset = []

print("开始调用先进大模型生成专家解释（知识蒸馏）...")
for idx, item in enumerate(sft_inputs):
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3", # 或者 qwen-max, gpt-4o 等
            messages=[
                {"role": "system", "content": "你是一个严格按照 JSON 格式输出的职业生涯规划专家。请务必使用中文（Chinese）来撰写 JSON 里的所有具体内容（Value），严禁在 JSON 字段值中使用英文进行长句解释。"},
                {"role": "user", "content": f"{item['instruction']}\n\n输入数据：{item['input']}"}
            ],
            temperature=0.3 # 降低随机性，保证 JSON 格式高度稳定
        )

        # 拿到大模型生成的完美 output（包含推荐理由、缺口和建议）
        llm_output = response.choices[0].message.content.strip()

        # 保存为标准的 Alpaca 格式
        final_sft_dataset.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": llm_output
        })

        if (idx + 1) % 10 == 0:
            print(f"已蒸馏完成 {idx + 1}/500 条专家数据...")

    except Exception as e:
        print(f"第 {idx} 条数据调用失败: {e}")
        continue

# 最终保存为训练大模型所需的 json 文件
with open("tech_advisor_sft_data.json", "w", encoding="utf-8") as f:
    json.dump(final_sft_dataset, f, ensure_ascii=False, indent=4)

print("🎉 恭喜！用于微调大模型的高质量数据集 tech_advisor_sft_data.json 已制作完成！")
