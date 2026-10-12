# UniMotorFM
### A Foundation Model for Electric Motor Fault Diagnosis and Condition Monitoring

**UniMotorFM** is a research project dedicated to developing transferable time-series foundation models for electric motor fault diagnosis and condition monitoring.

Our goal is to move beyond motor-specific models toward a unified representation that can generalize across different motors and operating conditions, while supporting diverse downstream tasks with limited labeled data or healthy data only.

## 🎯 Research Vision

Electric motor signals vary with motor design, operating conditions, sensor configurations, and degradation states. A key challenge is to learn representations that preserve condition-relevant information while remaining transferable across different motor systems.

We investigate three interconnected directions:

- **Data-efficient transfer:** Adapting pretrained time-series models to new motors with limited labeled data.
- **Transferable spectral knowledge:** Learning frequency-aware representations that capture reusable fault information across different motor characteristics.
- **Health assessment with healthy data:** Exploring healthy-data-based adaptation and regression-oriented approaches for motor condition assessment, reducing reliance on labeled fault data.

Together, these directions contribute toward a general-purpose foundation model for motor diagnosis and health monitoring.

## 🔬 Research Highlights

### 1. Data-Efficient Cross-Motor Fault Diagnosis

*A Novel Pre-training and Fine-tuning Paradigm for Data-Efficient Cross-Motor Fault Diagnosis*

Investigates how pre-training and fine-tuning can improve fault diagnosis on target motors when labeled data are scarce.

### 2. Transferable Spectral Priors

*Learning Transferable Spectral Priors for Cross-Motor Fault Diagnosis with Time-Series Foundation Model*

Studies how transferable spectral priors can help foundation models capture fault-related information across motors with different physical and frequency characteristics.

### 3. Healthy-Data-Based Condition Assessment

Explores adapting pretrained representations using healthy operating data and formulating condition assessment as a regression-oriented problem. This direction aims to support practical monitoring scenarios where labeled degradation or fault data are difficult to obtain.

## 🧭 Roadmap

- [x] Investigate data-efficient cross-motor fault diagnosis.
- [x] Explore transferable spectral priors for cross-motor generalization.
- [x] Develop a healthy-data-based adaptation approach for regression-oriented condition assessment.
- [ ] Unify datasets, model interfaces, and evaluation protocols across motor systems.
- [ ] Extend evaluation to diverse operating conditions and unseen motor domains.
- [ ] Improve reproducibility and prepare reusable training, adaptation, and inference pipelines.
- [ ] Investigate online health monitoring and continuous degradation assessment.

## 📦 Data and Resources

- **Pre-training dataset:** [PMSM Current and Vibration Dataset](https://huggingface.co/datasets/dydyd/current_vibration)
- **Code and models:** Training scripts, configurations, checkpoints, and evaluation tools will be organized as the project evolves.

## 🗓️ Updates

Record major releases, experiments, and research milestones here.

- **[YYYY-MM-DD] — [Update title]**
  - Added: New model, dataset, or functionality.
  - Improved: Training, transfer, or evaluation pipeline.
  - Results: Key findings and relevant experiment details.

## 📌 Project Status

UniMotorFM is an evolving research initiative. Its long-term objective is to establish a unified and transferable foundation-model framework for electric motor fault diagnosis, degradation assessment, and condition monitoring.
