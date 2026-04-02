# Competence-Aware Reasoning Efficiency (CARE)

<p align="center">
  <img src="https://img.shields.io/badge/Task-Video_Reasoning_Optimization-C92D39" alt="Task">
  <img src="https://img.shields.io/badge/Framework-Reinforcement_Learning-C92D39" alt="Framework">
  <img src="https://img.shields.io/badge/Model-Video--CARE--7B-007EC6" alt="Model">
</p>

<div align='center'>

!\[CARE Framework Overview]\(./Assets/care\_overview\.png null)

</div>

***

## 🔥 News

- **`2026.01.19`** 📝 Initial repository setup with release plan and reproducibility checklist.
- **`2026.01.15`** 🎯 Framework documentation for the CARE methodology published.

***

## 🎯 CARE: Competence-Aware Reasoning Efficiency

**CARE** is a dynamic reinforcement learning framework designed to optimize the cognitive budget and reasoning density of Multimodal Large Language Models (MLLMs) via competence-aware reward routing.

#### 👇 CARE addresses the following core challenges in multimodal video reasoning:

- 📊 **Competence Adaptation**: Maintains a smoothed competence estimate via exponential moving average of pass rates, routing training into progressive stages.
- ⚖️ **Balanced Reward**: Shifts reward preference from exploration-oriented long-form reasoning to efficiency-oriented concise reasoning as the model evolves.
- 📐 **Length Normalization**: Normalizes reasoning effort with batch-level statistics to avoid conflating verbosity with intrinsic task complexity.
- 🏆 **Posterior Amplification**: Strengthens reward signals for unexpectedly strong performance on historically difficult samples.

**Note:** To maintain double-blind anonymity during the peer review process for **IEEE Transactions on Image Processing (TIP)**, no source code is included in this branch.

## 🚀 Key Features

- **Adaptive Reasoning Length**: CARE exhibits a characteristic inverted-U evolution of reasoning length during training
- **Token Efficiency**: Yields shorter yet more informative reasoning traces at convergence
- **Seamless Integration**: Proposed mechanism is integrated into GRPO training pipeline with no additional inference-time overhead
- **Consistent Improvements**: Demonstrates improved reasoning accuracy and stabilized reinforcement learning across multiple benchmarks

***

## 📋 Release Plan

We are committed to open science and will release the full implementation upon acceptance.

**Camera-ready:**

- Inference code for the Video-CARE-7B model.
- Evaluation scripts for spatial-temporal video reasoning benchmarks.
- Visualization tools for reasoning length distributions (KDE) and Token Reduction Rate (Efficiency ROI) analysis.

**Within 30 days after online publication:**

- Full training codebase (based on TRL/DeepSpeed) integrating the CARE framework.
- Modular implementations of the Competence Monitor (EMA), Dynamic Reward Router, Posterior Amplifier, and Stabilizer Floor.
- Pretrained model checkpoints and training logs.
- A specific git tag will be created to match the exact version cited in the published paper.

***

## 🔬 Reproducibility

To ensure the results reported in the paper are rigorously reproducible, the future code release will include:

- **Environment:** An `environment.yml` and `requirements.txt` will be provided, specifying exact versions for PyTorch, Transformers, vLLM, and TRL.
- **Determinism:** Specific random seeds, hardware configurations (e.g., **8 × NVIDIA L20 48GB** setups), and DeepSpeed ZeRO-3 config files used for the reported experiments will be fully documented to prove resource-efficiency.
- **Configurations:**
  - Hyperparameters for the CARE routing mechanism (e.g., EMA momentum γ, Phase Thresholds T, Anchor a).
  - Dynamic Modulation parameters: Base coefficients B and weighting factors α, β.
  - Length Calibration configurations: Stabilizer floor L\_floor, Tolerance multiplier ω.
  - GRPO training configurations (e.g., Learning rate, Group size G).
- **Commands:** One-line shell scripts to reproduce the main reasoning performance results and the token efficiency ablation studies.

***

## 📊 Evaluation Benchmarks

Extensive experiments are conducted on multiple video reasoning and general video understanding benchmarks:

- VSI-Bench
- VideoMMMU
- MMVU (MC)
- MVBench
- TempCompass
- Video-MME

***

## 📊 Evaluation Results

### Main Performance Comparison

| Models                | Frames | VSI-Bench | VideoMMMU | MMVU(mc) | MVBench  | TempCompass | VideoMME(wo sub) |
| --------------------- | ------ | --------- | --------- | -------- | -------- | ----------- | ---------------- |
| LLaMA-VID             | -      | -         | -         | -        | 41.9     | 45.6        | -                |
| VideoLLaMA2           | -      | -         | -         | 44.8     | 54.6     | -           | 47.9             |
| LongVA-7B             | -      | 29.2      | 23.9      | -        | -        | 56.9        | 52.6             |
| VILA-1.5-8B           | -      | 28.9      | 20.8      | -        | -        | 58.8        | -                |
| VILA-1.5-40B          | -      | 31.2      | 34.0      | -        | -        | -           | 60.1             |
| Video-UTR-7B          | -      | -         | -         | -        | 58.8     | 59.7        | 52.6             |
| LLaVA-OneVision-7B    | -      | 32.4      | 33.8      | 49.2     | 56.7     | -           | 58.2             |
| Kangeroo-8B           | -      | -         | -         | -        | 61.1     | 62.5        | 56.0             |
| Qwen2.5-VL-7B         | -      | -         | 47.4      | 61.3     | 59.4     | 69.2        | 52.8             |
| Video-R1-7B           | 16     | 30.3      | 47.2      | 63.5     | 62.4     | 70.8        | 54.3             |
| DeepVideo-R1          | -      | 33.0      | 40.7      | 59.0     | 49.6     | 63.1        | 51.1             |
| TinyLLaVA-Video-R1    | 16     | -         | -         | 46.9     | -        | 49.5        | 46.6             |
| VIDEORFT              | 32     | -         | -         | 51.1     | 62.1     | -           | -                |
| Temporal-RLT          | 32     | -         | -         | 65.0     | -        | -           | 57.6             |
| Video-COM             | -      | -         | 50.2      | 65.4     | -        | 71.3        | -                |
| VideoChat-R1          | 128    | 33.0      | 52.0      | 64.8     | 63.6     | 74.5        | 64.1             |
| VideoChat-R1.5        | 128    | 36.1      | 50.0      | 67.0     | 65.7     | 73.9        | 64.8             |
| **Video-CARE (Ours)** | **16** | 33.9      | 50.2      | 64.2     | 64.4     | 73.1        | 57.3             |
| **Video-CARE (Ours)** | **32** | 34.3      | 51.0      | 64.5     | 65.0     | 73.6        | 60.6             |
| **Video-CARE (Ours)** | **64** | **36.2**  | **51.1**  | **64.8** | **65.7** | **73.8**    | **62.3**         |

### Key Highlights

- **Superior Token Efficiency**: Video-CARE achieves competitive or better performance using only **16 frames** compared to methods using **128 frames** (e.g., VideoChat-R1)
- **Best-in-class VideoMME**: With **62.3%** on VideoMME (wo sub), Video-CARE sets a new state-of-the-art among compared methods
- **Consistent Improvements**: Across all benchmarks, Video-CARE demonstrates consistent improvements over baseline Qwen2.5-VL-7B

***

## 📂 Data

This project utilizes public video datasets and synthesized reasoning trajectories but does not redistribute the raw videos directly.

Instructions and data-preparation scripts will be provided for:

**Training Datasets:**

- Video-R1-260k (Filtered and augmented with visual-grounding queries)

Users are required to follow the original licenses and usage terms of the respective datasets.

***

## 🔧 Third-party Components

The CARE framework builds upon or utilizes the following external projects/models:

- **Qwen2.5-VL-7B-Instruct:** The backbone Video-MLLM used for policy initialization.
- **TRL (Transformer Reinforcement Learning):** The base library for the GRPO algorithm implementation.
- **vLLM:** Utilized for high-throughput trajectory generation during the RL rollout phase.

Their licenses and usage terms remain with the original authors. This repository will include wrappers, dynamic reward functions, and adaptation code only.

***

## 📜 License and Intended Use

During the review period, this repository is intended solely for research reproducibility and verification by IEEE TIP reviewers and meta-reviewers.

The final license (e.g., Apache 2.0 or MIT) for our codebase will be determined and posted upon the official public release.

***

## 📚 Citation

A `CITATION.cff` file and a BibTeX entry for the paper will be added here upon acceptance.

***

## ✉️ Contact

For any questions regarding the paper, methodology, or this repository during the double-blind review process, please open an Issue in this repository. We will respond promptly while strictly maintaining author anonymity.
