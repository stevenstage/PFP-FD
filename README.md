# UniMotorFM
### Toward a Unified Foundation Model for Electric Motor Diagnosis and Condition Monitoring

**UniMotorFM** is an ongoing research initiative aiming to develop a unified time-series foundation model for electric motor fault diagnosis, degradation assessment, and condition monitoring.

Our goal is to learn transferable representations across different motors, operating conditions, and signal modalities, enabling diverse monitoring tasks with limited labeled data or healthy data only.

## 🔬 Research Foundations

### 1. Data-Efficient Motor Condition Monitoring with Time Series Foundation Models

[IEEE Xplore](https://ieeexplore.ieee.org/document/11517099)

Investigates the potential of time-series foundation models for data-efficient motor condition monitoring and downstream adaptation.

### 2. Learning Transferable Spectral Priors for Cross-Motor Fault Diagnosis with Time-Series Foundation Model

Explores transferable spectral priors to improve cross-motor fault diagnosis and understand how fault-related knowledge generalizes across different motor characteristics.

### 3. Healthy-Data-Based Condition Assessment

Investigates fine-tuning with healthy operating data and regression-oriented approaches to assess motor condition, reducing reliance on labeled fault and degradation data.

## 🚀 Current Project: A Unified Motor Foundation Model

Building on these research directions, we aim to develop a common foundation model that can support:

- **Cross-motor transfer** across different motor types and system configurations.
- **Operating-condition generalization** under changing speeds, loads, and other operating conditions.
- **Data-efficient adaptation** to new motors with limited labeled data.
- **Degradation assessment** through continuous severity estimation and regression.
- **Healthy-data-based monitoring** for practical scenarios where fault labels are unavailable.

## 📚 Dataset Collection and Statistics

**Current focus: collecting, cataloging, and characterizing approximately 40 candidate datasets.**

Rather than simply increasing the number of datasets, we aim to understand their diversity and suitability for pre-training and systematic generalization evaluation.

The ongoing dataset inventory tracks:

- Motor and system types, including relevant inverter-based systems.
- Signal modalities, sampling rates, and data volume.
- Operating-condition coverage, such as speed and load.
- Fault categories and continuous degradation severity, where available.
- Healthy-data availability and potential downstream tasks.
- Data accessibility, licensing, and preprocessing requirements.

Existing resource: [PMSM Current and Vibration Dataset](https://huggingface.co/datasets/dydyd/current_vibration)

The candidate dataset count reflects the collection and screening process; it does not imply that all datasets have already been integrated into pre-training.

### Dataset Inventory Template

| Dataset | System / Motor | Signals | Operating Conditions | Fault / Severity | Status |
|---|---|---|---|---|---|
| Dataset 01 | TBD | TBD | TBD | TBD | Under review |
| Dataset 02 | TBD | TBD | TBD | TBD | Under review |
| Dataset 03 | TBD | TBD | TBD | TBD | Under review |

*The full inventory can be maintained separately as the collection grows.*

## ✅ TODO

### Dataset Collection and Standardization
- [x] Establish the initial research direction for cross-motor diagnosis and condition monitoring.
- [ ] Continue collecting and screening candidate datasets.
- [ ] Complete dataset statistics and metadata documentation.
- [ ] Evaluate data quality, operating-condition coverage, and degradation information.
- [ ] Define consistent preprocessing and data-split protocols.

### Unified Foundation Model
- [ ] Establish a unified multi-dataset pre-training pipeline.
- [ ] Investigate transfer across motors, operating conditions, and signal modalities.
- [ ] Develop consistent evaluation protocols for cross-motor generalization.
- [ ] Integrate fault diagnosis, regression-based degradation assessment, and healthy-data-based monitoring.
- [ ] Release reproducible training, fine-tuning, and inference tools.

## 🗓️ Updates

A chronological record of dataset curation, model development, experiments, and releases.

### [2026-10] — Dataset Collection and Characterization
- **Data:** Ongoing collection and statistical analysis of approximately 40 candidate datasets.
- **Curation:** Organizing metadata on motor types, signal modalities, operating conditions, fault severity, and healthy-data availability.
- **Research:** Preparing the data foundation for a unified motor foundation model.

### [YYYY-MM-DD] — Update Title
- **Added:** New datasets, models, or functionality.
- **Improved:** Data processing, training, or evaluation pipeline.
- **Experiments:** Newly completed validation or benchmark results.
- **Next:** Planned work for the next development stage.

## 📖 Citation

**Data-Efficient Motor Condition Monitoring with Time Series Foundation Models**

[IEEE Xplore](https://ieeexplore.ieee.org/document/11517099)

Further publications, source code, pretrained checkpoints, and benchmark results will be added as they become available.
