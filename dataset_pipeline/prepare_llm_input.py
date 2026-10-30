# prepare_llm_input.py
import pandas as pd

def merge_and_clean_requirements(csv_path, output_txt_path):
    # 1. 读取 A 爬取好的 CSV 数据
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk') # 兼容可能出现的编码问题

    # 2. 提取 requirements 列并去除空值
    if "requirements" not in df.columns:
        print("错误：CSV 文件中没有发现 'requirements' 列！")
        return

    reqs = df["requirements"].dropna()

    # 3. 【核心优化】：去除完全重复的要求文本
    # 很多相同大厂的岗位要求是一模一样的，去重能帮大模型省下大量无用字符，防止其输出时“疲劳遗漏”
    unique_reqs = reqs.drop_duplicates()

    # 4. 写入 txt 文件，并加上序号，方便大模型逐条分析
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for idx, req in enumerate(unique_reqs, 1):
            cleaned_req = str(req).replace('\n', ' ').strip()
            f.write(f"【岗位{idx}】: {cleaned_req}\n\n")

    print(f"合并完成！原始数据共 {len(reqs)} 条，去重后精简为 {len(unique_reqs)} 条独特要求。")
    print(f"已成功生成大模型输入文件：{output_txt_path}")

if __name__ == "__main__":
    # 请确保 jobs.csv 放在同一目录下，或修改为实际路径
    merge_and_clean_requirements("all.csv", "merged_requirements.txt")