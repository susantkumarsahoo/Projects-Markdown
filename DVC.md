# **DVC – Complete Command-Line Reference (Enterprise Ready)**

This document provides a comprehensive command-line reference for using **DVC (Data Version Control)** across production-grade machine learning workflows.  
Commands are organized into five primary operational categories.

---

## **1. Initialization & Repository Setup**

| Purpose | Command |
|--------|---------|
| Initialize DVC | `dvc init` |
| Initialize without Git | `dvc init --no-scm` |
| Add remote storage | `dvc remote add -d myremote s3://bucket/path` |
| Configure remote access | `dvc remote modify myremote access_key_id <KEY>` |
| Check version | `dvc --version` |

---

## **2. Data & Model Tracking**

| Operation | Command |
|-----------|---------|
| Track a dataset | `dvc add data/raw.csv` |
| Track a directory | `dvc add data/` |
| Force-update a tracked directory | `dvc add --force data/` |
| Track model artifacts | `dvc add models/model.pkl` |
| Remove tracked file | `dvc remove data/raw.csv.dvc` |
| Commit workspace changes | `dvc commit` |

---

## **3. Pipelines, Stages & Automation**

| Purpose | Command |
|---------|---------|
| Create pipeline stage | `dvc stage add -n preprocess -d src/preprocess.py -d data/raw.csv -o data/clean.csv python src/preprocess.py` |
| Run a pipeline | `dvc repro` |
| Visualize pipeline DAG | `dvc dag` |
| Freeze stage | `dvc freeze preprocess` |
| Unfreeze stage | `dvc unfreeze preprocess` |
| Check parameter differences | `dvc params diff` |

---

## **4. Experiments (Experiment Management & MLOps)**

| Purpose | Command |
|---------|---------|
| Run an experiment | `dvc exp run` |
| Run with parameter override | `dvc exp run -S train.lr=0.001` |
| List experiments | `dvc exp show` |
| Compare experiments | `dvc exp diff` |
| Apply experiment results | `dvc exp apply <exp_id>` |
| Create Git branch from experiment | `dvc exp branch <exp_id> exp_branch` |
| Push experiments to remote | `dvc exp push myremote` |
| Pull experiments from remote | `dvc exp pull myremote` |
| Remove experiments | `dvc exp remove <exp_id>` |

---

## **5. Remote Storage Operations**

| Action | Command |
|--------|---------|
| Push data/models to remote | `dvc push` |
| Pull data/models from remote | `dvc pull` |
| Fetch data without modifying workspace | `dvc fetch` |
| Set default remote | `dvc remote default myremote` |
| Remove remote | `dvc remote remove myremote` |

---

## **Additional High-Value Commands**

### **Metrics & Parameters**

| Command | Purpose |
|---------|---------|
| `dvc metrics show` | Display metrics |
| `dvc metrics diff` | Compare metrics |
| `dvc params diff` | Compare parameters |

---

### **Cache, Lock & Diagnostics**

| Command | Purpose |
|---------|---------|
| `dvc checkout` | Restore file state |
| `dvc status` | Check for changes |
| `dvc doctor` | Run diagnostics |
| `dvc gc -w` | Clean unused cache |

---

### **Import & Export**

| Command | Purpose |
|---------|---------|
| `dvc import <url> <path>` | Import from remote repo |
| `dvc import-url` | Import external dataset |
| `dvc update` | Update imported data |
| `dvc get <repo> data/file.csv` | Download without tracking |

---

## **End-to-End Pipeline Workflow (Reference)**

```sh
git init
dvc init

dvc remote add -d storage s3://mlbucket/artifacts

dvc add data/raw.csv
git add data/raw.csv.dvc .gitignore
git commit -m "Added raw dataset"

dvc stage add -n preprocess \
    -d src/preprocess.py \
    -d data/raw.csv \
    -o data/clean.csv \
    python src/preprocess.py

dvc repro
dvc push
