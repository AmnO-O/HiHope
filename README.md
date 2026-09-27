# Linguistic Analysis & Text Evaluation Framework

A research workbench and experimentation framework for multilingual text evaluation, contextual sequence modeling, and representation analysis in NLP.

## Overview

This repository provides tools and utilities for training, evaluating, and analyzing contextual regression models across multilingual corpora.

### Key Capabilities
- **Modular Pipeline:** Flexible configuration-driven training and evaluation workflows.
- **Multilingual Support:** Compatible with modern pretrained transformer backbones.
- **Continuous Metric Evaluation:** Supports regression metrics, correlation analysis (Spearman, Pearson), and distribution estimation.
- **Experiment Tracking:** Logging, validation checkpointing, and customizable training stages.

## Requirements & Setup

### Environment
Ensure Python 3.9+ and PyTorch are installed in your environment:

```bash
pip install -r requirements.txt
```

### Quick Run

To execute the training pipeline using the default configuration:

```bash
python run.py --config config/default.json
```

Custom parameter overrides can be supplied via the command line:

```bash
python run.py --config config/default.json --set batch_size=16 --set freeze_epochs=3 --set unfreeze_epochs=9
```

## Notebooks

For interactive analysis and exploratory evaluation, refer to the provided notebook in `notebooks/kaggle_run.ipynb`.

## License

MIT License
