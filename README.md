# Competence-Aware Reasoning Efficiency (CARE)

This repository documents the release plan and reproducibility checklist for **CARE**, a dynamic reinforcement learning framework designed to optimize the cognitive budget and reasoning density of MLLMs via competence-aware reward routing.

**Note:** To maintain double-blind anonymity during the peer review process for **IEEE Transactions on Image Processing (TIP)**, no source code is included in this branch.

## Release Plan

We are committed to open science and will release the full implementation upon acceptance.

**Camera-ready:**
* Inference code for the Video-CARE-7B model.
* Evaluation scripts for spatial-temporal video reasoning benchmarks.
* Visualization tools for reasoning length distributions (KDE) and Token Reduction Rate (Efficiency ROI) analysis.

**Within 30 days after online publication:**
* Full training codebase (based on TRL/DeepSpeed) integrating the CARE framework.
* Modular implementations of the Competence Monitor (EMA), Dynamic Reward Router, Posterior Amplifier, and Stabilizer Floor.
* Pretrained model checkpoints and training logs.
* A specific git tag will be created to match the exact version cited in the published paper.

## Reproducibility

To ensure the results reported in the paper are rigorously reproducible, the future code release will include:

* **Environment:** An `environment.yml` and `requirements.txt` will be provided, specifying exact versions for PyTorch, Transformers, vLLM, and TRL.
* **Determinism:** Specific random seeds, hardware configurations (e.g., **8 $\times$ NVIDIA L20 48GB** setups), and DeepSpeed ZeRO-3 config files used for the reported experiments will be fully documented to prove resource-efficiency.
* **Configurations:**
  * Hyperparameters for the CARE routing mechanism (e.g., EMA momentum $\gamma$, Phase Thresholds $\mathcal{T}$, Anchor $a$).
  * Dynamic Modulation parameters: Base coefficients $\mathcal{B}$ and weighting factors $\alpha$, $\beta$.
  * Length Calibration configurations: Stabilizer floor $L_{floor}$, Tolerance multiplier $\omega$.
  * GRPO training configurations (e.g., Learning rate, Group size$\G$).
* **Commands:** One-line shell scripts to reproduce the main reasoning performance results and the token efficiency ablation studies.

## Data

This project utilizes public video datasets and synthesized reasoning trajectories but does not redistribute the raw videos directly. 

Instructions and data-preparation scripts will be provided for:

**Training Datasets:**
* Video-R1-260k (Filtered and augmented with visual-grounding queries)

**Evaluation Benchmarks:**
* VSI-Bench
* VideoMMMU
* MMVU (MC)
* MVBench
* TempCompass
* Video-MME

Users are required to follow the original licenses and usage terms of these respective datasets.

## Third-party Components

The CARE framework builds upon or utilizes the following external projects/models:

* **Qwen2.5-VL-7B-Instruct:** The backbone Video-MLLM used for policy initialization.
* **TRL (Transformer Reinforcement Learning):** The base library for the GRPO algorithm implementation.
* **vLLM:** Utilized for high-throughput trajectory generation during the RL rollout phase.

Their licenses and usage terms remain with the original authors. This repository will include wrappers, dynamic reward functions, and adaptation code only.

## License and Intended Use

During the review period, this repository is intended solely for research reproducibility and verification by IEEE TIP reviewers and meta-reviewers.

The final license (e.g., Apache 2.0 or MIT) for our codebase will be determined and posted upon the official public release.

## Citation

A `CITATION.cff` file and a BibTeX entry for the paper will be added here upon acceptance.

## Contact

For any questions regarding the paper, methodology, or this repository during the double-blind review process, please open an Issue in this repository. We will respond promptly while strictly maintaining author anonymity.
