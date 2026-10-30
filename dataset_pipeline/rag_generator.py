# src/rag_generator.py
import os
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化大模型客户端
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

def generate_recommendation_analysis(user_profile: str, job_info: dict) -> str:
    """
    调用大语言模型，生成推荐理由和技能缺口分析。
    """
    # 构造 System Prompt：定义 AI 的角色和强制输出格式
    system_prompt = """
    你是一个资深的校招求职辅导专家与大厂HR。你的任务是根据用户的简历画像和具体的岗位JD，进行深度的匹配度分析。

    【核心要求】
    请严格按照以下 JSON 格式输出结果，不要输出任何额外的 Markdown 标记（如 ```json ）或解释性文字：
    {
        "recommendation_reason": "结合用户背景与岗位要求的推荐理由（100字左右）",
        "current_advantages": ["优势1", "优势2"],
        "skill_gaps": ["缺口1", "缺口2"],
        "improvement_advice": "针对技能缺口的具体背景提升建议（如需要学习什么具体的工具或算法）",
        "match_score_explanation": "从专业、技能、软素质三个维度的匹配度总结"
    }
    """

    # 构造 User Prompt：注入 B 同学检索出的数据和用户真实画像
    user_prompt = f"""
    【用户画像】
    {user_profile}

    【推荐岗位信息】
    岗位名称：{job_info['job_title']}
    所属公司：{job_info['company']}
    岗位要求 (Requirements)：{job_info['requirements']}
    岗位描述 (JD Text)：{job_info['jd_text']}

    请根据上述信息，生成结构化的分析报告。
    """

    print("🚀 正在请求大模型生成分析报告，请稍候...")

    try:
        # 调用大模型 API
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3", # 请确保这里与你使用的平台要求的名称严格一致
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    # ====== 下面是新增的精细化错误排查模块 ======
    except openai.BadRequestError as e:
        print(f"❌ [400 错误] 参数不正确或模型不存在。")
        print(f"   详细报错: {e.response.json() if hasattr(e, 'response') else str(e)}")
        return json.dumps({"error": "400 Bad Request", "message": str(e)})

    except openai.AuthenticationError as e:
        print(f"❌ [401 错误] API Key 未正确设置或无效。")
        print(f"   详细报错: {str(e)}")
        return json.dumps({"error": "401 Unauthorized", "message": str(e)})

    except openai.PermissionDeniedError as e:
        print(f"❌ [403 错误] 余额不足、未实名认证或权限不够。")
        print(f"   详细报错: {str(e)}")
        return json.dumps({"error": "403 Forbidden", "message": str(e)})

    except openai.RateLimitError as e:
        print(f"❌ [429 错误] 触发并发限流 (Rate Limits)，请稍后重试。")
        print(f"   详细报错: {str(e)}")
        return json.dumps({"error": "429 Rate Limit", "message": str(e)})

    except openai.InternalServerError as e:
        print(f"❌ [500/503 错误] 服务器负载过高或发生未知错误。")
        print(f"   详细报错: {str(e)}")
        return json.dumps({"error": "500 Server Error", "message": str(e)})

    except openai.APIConnectionError as e:
        print(f"❌ [网络错误] 无法连接到服务器，请检查 BASE_URL 或本地网络。")
        print(f"   详细报错: {str(e)}")
        return json.dumps({"error": "Connection Error", "message": str(e)})

    except Exception as e:
        print(f"❌ [未知错误] 发生了预料之外的异常。")
        print(f"   详细报错: {str(e)}")
        return json.dumps({"error": "Unknown Error", "message": str(e)})

# 在文件最上方记得导入 B 同学的画像（如果报错找不到，可以 import sys; sys.path.append(...) 解决路径问题）
from user_profile import user_profile

def main():
    # =========================================================
    # 1. 动态读取并转换用户画像 (核心更新点)
    # =========================================================
    # 大模型更擅长理解自然语言，所以我们把 B 同学的字典拼接成一段完整的话
    user_profile_text = (
        f"用户画像：姓名{user_profile['name']}，目前是{user_profile['major']}专业的{user_profile['grade']}（{user_profile['education']}）。"
        f"掌握的技术技能包括：{', '.join(user_profile['skills'])}。"
        f"意向工作城市是{user_profile['target_city']}，"
        f"感兴趣的求职方向为：{', '.join(user_profile['interested_categories'])}，"
        f"最核心的职业偏好是【{user_profile['career_preference']}】。"
    )

    print(f"👤 当前加载的用户画像：\n{user_profile_text}\n")

    # =========================================================
    # 2. 读取 B 同学传来的 Top-K 推荐结果
    # =========================================================

    # 自动获取当前 rag_generator.py 所在的文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 拼装出完整的 json 文件路径
    # 3. 绝对路径加载 B 同学传来的 Top-K 推荐结果 (指向 outputs 文件夹)

    input_file = os.path.join(current_dir, "outputs", "topk_recommendations.json")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            top_jobs = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {input_file}，请检查路径。")
        return
    final_results = []
    print(f"🚀 开始批量处理 B 同学推荐的 {len(top_jobs)} 个岗位...")
    print("-" * 50)

    # =========================================================
    # 3. 循环遍历，逐个生成 LLM 分析报告
    # =========================================================
    for i, job in enumerate(top_jobs):
        print(f"正在分析 [{i+1}/{len(top_jobs)}]: {job['job_title']} - {job['company']}")

        job_info_for_llm = {
            "job_title": job["job_title"],
            "company": job["company"],
            "requirements": "请从下方的 JD 中自行提取并分析技能缺口",
            "jd_text": job["jd_text"]
        }

        # 传入动态拼接好的 user_profile_text
        analysis_str = generate_recommendation_analysis(user_profile_text, job_info_for_llm)

        try:
            analysis_json = json.loads(analysis_str)
            job["llm_analysis"] = analysis_json
        except Exception as e:
            print(f"❌ 岗位 {job['job_id']} 返回格式解析失败: {e}")
            job["llm_analysis"] = {"error": "JSON解析失败", "raw_output": analysis_str}

        final_results.append(job)
        print(f"✅ [{job['job_title']}] 分析完成！\n")

    # =========================================================
    # 4. 保存最终结果
    # =========================================================
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 最终完整的文件路径
    output22_file = os.path.join(output_dir, "final_recommendations_with_llm.json")

    with open(output22_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("-" * 50)
    print(f"🎉 全部处理完毕！最终数据已打包保存至: {output22_file}")

if __name__ == "__main__":
    main()
