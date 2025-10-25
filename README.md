# A Novel Pre-training and Fine-tuning Paradigm for Data-Efficient Cross-Motor Fault Diagnosis
Official implemental code for the paper "A Novel Pre-training and Fine-tuning Paradigm for Data-Efficient Cross-Motor Fault Diagnosis"

## Pre-training Dataset    
### Falut Diagnosis
The falut diagnosis raw dataset was collected and arranged by the [Korea Advanced Institute of Science and Technology](https://www.sciencedirect.com/science/article/pii/S2352340923000707), and the adaptation edition used in this work can be found in [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/dydyd/current_vibration)

__including:__  
- 1.0 kW, 1.5 kW and 3.0kW PMSM

Total __576M__ and __147M__ time points for current signal and vibration signal respectively


Total __1.9B__ observations 

#### Details   

| Parameters                | 1st PMSM (1.0 kW) | 2nd PMSM (1.5 kW) | 3rd PMSM (3.0 kW) | Unit   |
|----------------------------|-------------------|--------------------|-------------------|--------|
| Manufacturing company       | Higen motors      | Higen motors       | Higen motors      | -      |
| Rated power                 | 1000              | 1500               | 3000              | Watt   |
| Input voltage               | 380               | 380                | 380               | AC Voltage |
| Frequency                   | 60                | 60                 | 60                | Hz     |
| Number of phase             | 3                 | 3                  | 3                 | Phase  |
| Number of pole              | 4                 | 4                  | 4                 | -      |
| Rated torque                | 3.18              | 4.77               | 9.55              | Nm     |
| Rated speed                 | 3000              | 3000               | 3000              | RPM    |
| Synchronous inductance      | 0.0               | 0.0                | 0.0               | H      |
| Magnetic flux               | 400               | 350                | 300               | mT     |
| Rotor inertia               | 2.07              | 7.48               | 14.34             | Kgm²   |
| Inter-turn resistance value (R_it) | 0.1385      | 0.0958            | 0.1087            | Ohm    |
| Inter-coil resistance value (R_cc) | 0.0409      | 0.3021            | 0.1534            | Ohm    |

## Fine-tuning Dataset   

## Key Result Comparison

Here are the results of SOTA deep learning models and fine-tuned foundation models for fault diagnosis.

The SOTA models are from [tsai](https://github.com/timeseriesAI/tsai):
- [XCM](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XCM.py) -
  An Explainable Convolutional Neural Network (Fauvel, 2021)
  ([paper](https://hal.inria.fr/hal-03469487/document))
- [TSPerceiver](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TSPerceiver.py) -
  Adapted from Perceiver IO (Jaegle, 2021)
  ([paper](https://arxiv.org/abs/2107.14795))
- [TSSequencerPlus](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TSSequencerPlus.py) -
  Adapted from Sequencer (Tatsunami, 2022)
  ([paper](https://arxiv.org/abs/2205.01972))
- [PatchTST](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/PatchTST.py) -  
  (Nie, 2022)([paper](https://arxiv.org/abs/2211.14730))  

The foundation models are from [Mantis](https://github.com/vfeofanov/mantis).
