# Momentum Contrast for Unsupervised Visual Representation Learning

This repository contains the implementation and experiments for our final project on Momentum Contrast (MoCo) applied to the CIFAR-10 dataset. The project demonstrates self-supervised learning principles, including contrastive learning, and evaluates MoCo's performance on small datasets.

## Project Overview
Momentum Contrast (MoCo) is a self-supervised learning framework designed to learn meaningful visual representations without labels. This project evaluates MoCo’s performance on CIFAR-10 and compares it with a fully supervised baseline.

## Contributions
### Team Members:
1. **Mohan Vamsi Krishna Yanamadala**:
   - Implemented MoCo architecture, and Conducted hyperparameter tuning for the CIFAR-10 dataset.
   - Wrote the training and evaluation pipelines for pretraining and linear evaluation.

2. **Gouthami Nadella**:
   - Designed and implemented data augmentation techniques for the TwoCropsTransform class.
   - Led the fine-tuning phase by configuring the experimental setup and analyzing the results.

3. **SivaSai Atchyut Akella**:
   - Managed dataset preparation and implemented the training loop for efficient pretraining.
   - Visualized and analyzed loss curves for performance monitoring.

### Important Changes:
- Enhanced MoCo for small datasets by reducing queue size and optimizing temperature.
- Conducted fine-tuning experiments to align MoCo with CIFAR-10-specific challenges.
