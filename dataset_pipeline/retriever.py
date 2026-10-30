# src/retriever.py
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载模型
model = SentenceTransformer('shibing624/text2vec-base-chinese')

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return float(dot_product / (norm_vec1 * norm_vec2))

def calculate_raw_similarity(user, job):
    """
    【文本净化优化】：
    不再使用“口水句”，直接拼接“核心特征词”，能极大减少背景噪音，提升区分度。
    返回：0.0 ~ 1.0 之间的原始相似度
    """
    # 1. 净化用户文本：只拼接高含金量标签
    user_features = [
        user['major'],
        user['education'],
        " ".join(user['skills']),
        user['target_city'],
        user['career_preference']
    ]
    user_text = " ".join(user_features)

    # 2. 净化岗位文本：拼接标题、城市和干净的要求、职责
    job_features = [
        job.get('job_title', ''),
        job.get('city', ''),
        job.get('requirements', ''),
        job.get('jd_text', '')
    ]
    job_text = " ".join(job_features)

    # 3. 计算向量
    user_vector = model.encode(user_text)
    job_vector = model.encode(job_text)

    # 4. 返回 0-1 之间的原始相似度
    similarity = calculate_cosine_similarity(user_vector, job_vector)
    return max(0.0, similarity)