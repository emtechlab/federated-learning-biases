# Federated-Learning-Bias-CaseStudy

[![DOI](https://img.shields.io/badge/DOI-10.1145/3560905.3568305-blue.svg)](https://doi.org/10.1145/3560905.3568305)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An in-depth case study analyzing bias in federated learning arising from label and **sampling feature** heterogeneity across edge devices. This repository provides the code to reproduce our experiments on CIFAR-10 and Cholec80, including data partitioning, FL training with normalization variants, evaluation, and visualization.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Installation & Prerequisites](#installation--prerequisites)  
3. [Data Preparation](#data-preparation)  
4. [Configuration](#configuration)  
5. [Usage](#usage)  
   - [Training](#training)  
   - [Evaluation](#evaluation)  
   - [Visualization](#visualization)  
6. [Experiments & Results](#experiments--results)  
7. [Citation](#citation)  

---

## Introduction

Federated Learning (FL) enables collaborative model training across edge devices without sharing raw data. However, **heterogeneous sensors** introduce two key sources of bias:

- **Label heterogeneity**: uneven class distributions across clients.  
- **Sampling feature heterogeneity**: differences in feature representations (e.g., noise levels) due to diverse device quality.  

Our empirical case study on CIFAR-10 and a surgical task dataset (Cholec80) shows that while normalization methods (BatchNorm, GroupNorm, InstanceNorm, LayerNorm) can improve overall performance, **none eliminate per-client bias induced by sampling feature heterogeneity**.

**Contributions**  
1. **Empirical analysis** of label vs. sampling feature heterogeneity impact on per-client bias in FL.  
2. **Evaluation** of state-of-the-art normalization techniques under both heterogeneity types.  
3. **Insights** into performance–fairness–resource trade-offs, motivating new bias-mitigation strategies for FL.

---

## Installation & Prerequisites

1. **Clone the repository**  
   ```bash
   git clone https://github.com/emtechlab/federated-learning-biases.git
   cd federated-learning-biases
