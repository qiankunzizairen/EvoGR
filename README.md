# EvoGR (Evo 2-based Genomic Representation)

This repository implements EvoGR, the method proposed in **Structure-aware transfer of pretrained genomic representations for population and cross-clade classification**.

## Model Architecture

EvoGR uses pretrained Evo 2 to extract genomic sequence representations. It combines local context and variant information to construct structure-aware genomic representations for population and cross-clade classification. The Evo 2 backbone remains frozen, while task-specific classification heads can be trained for each downstream task.

![EvoGR model architecture](fig/framework.png)

## Quick Start

### 1. Install dependencies

We recommend Python 3.10 or later and a Linux environment with CUDA acceleration for Evo 2 inference.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the source paths:

```bash
export PYTHONPATH="$PWD/population/src:$PWD/cross_clade/src:$PYTHONPATH"
```
### 2. Download the Evo 2 checkpoint

Download link: https://huggingface.co/arcinstitute/evo2_7b_base/resolve/main/evo2_7b_base.pt

### 3. Population classification

After preparing the input data, Evo 2 checkpoint, and configuration file, run the full pipeline:

```bash
python population/scripts/run_pipeline.py --config configs/base.yaml
```

Data validation, sequence construction, Evo 2 representation extraction, cross-validation, and result validation can also be run separately using the scripts in `population/scripts/`.

### 4. Cross-clade classification

After preparing the cross-clade configuration and input data, run cross-validation:

```bash
python cross_clade/scripts/run_cv.py \
  --head transformer \
  --config configs/experiment.yaml
```
