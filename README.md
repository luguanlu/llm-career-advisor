# LLM Career Advisor

An end-to-end large language model project for Chinese campus-recruitment job recommendation. The project collects and cleans job descriptions, scores candidate–job matches with a hybrid retrieval pipeline, creates supervised fine-tuning (SFT) data, fine-tunes instruction-following models with LoRA, and compares training strategies through ablation experiments.

The repository documentation, directory names, and setup instructions are in English. Chinese appears only where it is intrinsic to the task: job descriptions, synthetic SFT examples, and prompts intended to produce Chinese career advice.

## Project pipeline

```text
Job-source notebooks
        |
        v
Cleaning + taxonomy + skill extraction
        |
        v
Hybrid candidate–job retrieval and ranking
        |
        v
Synthetic Chinese SFT dataset generation
        |
        v
LoRA fine-tuning with LLaMA-Factory
        |
        v
Hyperparameter, data-scale, and NEFTune ablations
```

## What is included

| Area | Contents |
| --- | --- |
| `data_collection/` | Jupyter notebooks for collecting job postings from three sources. |
| `data_processing/` | Cleaning, job taxonomy, skill extraction, and descriptive-statistics scripts. |
| `dataset_pipeline/` | Candidate-profile generation, rule/embedding-based matching, Top-K retrieval, and LLM-assisted SFT-data creation. |
| `training/` | Project-specific SFT train/predict scripts, dataset registry, and train/test splits. |
| `experiments/` | BOHB-style configuration search, data-scale experiments, Taguchi configurations, result tables, and figures. |
| `ablation/` | NEFTune comparison configurations, metrics, and saved plots. |
| `docs/project_report.pdf` | Project report supplied with the original project package. |

Raw scraped job postings, model checkpoints, logs, caches, and API credentials are intentionally excluded. The tracked SFT data is the curated project dataset used by the training configurations.

## Requirements

- Python 3.10 or later
- CUDA-capable GPU for training
- A provider API key only when regenerating LLM-written SFT responses
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning and evaluation

Create a Python environment for the data and retrieval pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use an LLM provider during SFT-data generation, keep secrets local:

```bash
cp .env.example .env
# Edit .env and set LLM_API_KEY. Never commit this file.
```

## Running the pipeline

### 1. Collect and prepare job data

Run the notebooks in `data_collection/scrapers/` only after confirming that your use complies with the relevant websites' terms, robots rules, and applicable law. Export the merged records locally, then run the scripts in `data_processing/` in this order:

```text
data_cleaning.py
taxonomy_labeling.py
skill_extraction.py
dataset_stats.py
```

The processing scripts preserve their original expected local paths (`data/all.csv` and `outputs/`). Create those directories locally or adapt the path constants before execution.

### 2. Produce ranked examples and SFT data

`dataset_pipeline/` combines a rule score with sentence-embedding cosine similarity. `create_top3.py` and `create_sft_dataset.py` call an OpenAI-compatible endpoint and require `LLM_API_KEY`; they now fail safely if the variable is absent.

```bash
cd dataset_pipeline
python create_top3.py
# or
python create_sft_dataset.py
```

These scripts require a local job CSV and write generated outputs locally. Review generated examples for quality and privacy before adding them to any dataset.

### 3. Fine-tune with LLaMA-Factory

Install LLaMA-Factory separately, following its official instructions. Then copy the project dataset directory and registry into its `data/` directory:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# From this repository's root:
cp -R training/data/. LLaMA-Factory/data/
cp training/dataset_info.json LLaMA-Factory/data/dataset_info.json
```

Before starting a run, update `dataset_dir`, output paths, model identifiers, GPU selection, and any platform-specific environment variables in `training/scripts/`. The provided scripts document the configurations used for GLM-4-9B, Qwen2-7B, and Llama-3-8B LoRA SFT.

## Experiments

- `experiments/bohb/`: 12 first-stage configurations and comparative result summaries.
- `experiments/data_scale/`: 30%, 60%, and 100% training-data experiments.
- `experiments/taguchi/`: L9 orthogonal-design configurations and main-effect summaries.
- `ablation/configs_and_scripts/stage2/`: baseline, NEFTune alpha 5, and NEFTune alpha 10 configurations with metrics and plots.

All training configurations assume local GPU resources. Replace local model paths before running them on another machine.

## Security and reproducibility notes

- API keys are read from environment variables, not source code.
- Do not commit `.env`, raw data, model weights, checkpoints, or logs.
- The project uses LLaMA-Factory as an external dependency rather than republishing it. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing context.
- Results in this repository are recorded experiment artefacts, not a guarantee of performance on a new dataset or hardware environment.

## License

No project license has been selected yet. Add a license before redistributing or accepting contributions.
