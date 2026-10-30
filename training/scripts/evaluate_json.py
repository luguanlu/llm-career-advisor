import json

predict_file = "./saves/Llama3-8B/lora/tech_advisor_predict/generated_predictions.jsonl"
total_count = 0
success_count = 0
failed_cases = []

with open(predict_file, "r", encoding="utf-8") as f:
    for line in f:
        total_count += 1
        data = json.loads(line)
        # LLaMA-Factory 的预测结果通常存在 "predict" 字段中
        prediction_text = data.get("predict", "")

        # 清理可能存在的 markdown 代码块符号 (比如 ```json 和 ```)
        clean_text = prediction_text.strip().removeprefix("```json").removesuffix("```").strip()

        try:
            # 尝试将回答解析为 JSON 字典
            parsed_json = json.loads(clean_text)

            # 检查是否严格包含了你要求的三个 Key
            if all(key in parsed_json for key in ["reason", "gap", "suggestion"]):
                success_count += 1
            else:
                failed_cases.append(f"第 {total_count} 题：格式正确，但缺少要求的 Key")
        except json.JSONDecodeError:
            failed_cases.append(f"第 {total_count} 题：JSON 解析失败！回答内容是：\n{prediction_text}")

print("================ 测试报告 ================")
print(f"总测试数据量: {total_count} 条")
print(f"JSON 完美解析且字段完整: {success_count} 条")
print(f"模型良率: {(success_count/total_count)*100:.2f}%")

if failed_cases:
    print("\n--- 失败案例诊断 ---")
    for case in failed_cases:
        print(case)
        print("-" * 20)