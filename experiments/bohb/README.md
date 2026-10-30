# BOHB Stage-1 Candidates for LLaMA3-8B (GPTQ + LoRA)

此目录包含为第一轮 BOHB 风格超参搜索准备的 12 个短训练候选配置（stage-1）。

- 目标模型：Meta-Llama-3-8B（GPTQ 权重）用 LoRA 微调。
- 约束：不启用 NEFTune 和 R-Drop（`neftune_noise_alpha: 0`, `rdrop_alpha: 0`）。
- 每个配置均为短训练（`max_steps: 100`），以快速评估超参表现。
- 文件名：`bohb_stage1_01.yaml` … `bohb_stage1_12.yaml`。
- 输出目录：每个配置使用不同的 `output_dir`，避免覆盖。

使用方式（示例）：
```bash
# 进入项目根
cd lgx-research/LLaMA-Factory

# 使用 CLI 启动某个候选
llamafactory-cli train --config experiments/bohb/bohb_stage1_01.yaml
```

注意：本目录仅新增配置文件，不会修改项目源码。请在运行前确认集群/accelerate 配置与资源。
