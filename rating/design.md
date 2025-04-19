# Rating Module Refactoring Design

## Current Structure Analysis

The current structure of the rating module has several issues:

1. **Inconsistent Path Handling**: Some scripts use hardcoded paths, while others use environment variables.
2. **Duplicate Code**: Training and sweep scripts have significant duplication.
3. **Scattered Model Implementations**: Model implementations are spread across different directories.
4. **Inconsistent Training Interfaces**: Different models have different training scripts with varying interfaces.

## Proposed Structure

```
rating/
├── models/
│   ├── __init__.py
│   ├── base.py           # Base model classes
│   ├── mf.py             # Matrix Factorization models
│   └── ncf.py            # Neural Collaborative Filtering models
├── trainers/
│   ├── __init__.py
│   ├── base.py           # Base trainer classes
│   ├── weighted.py       # Weighted trainer
│   └── ncf.py            # NCF trainer
├── datasets/
│   ├── __init__.py
│   ├── base.py           # Base dataset classes
│   ├── factory.py        # Dataset factory
│   └── cold_start.py     # Cold start dataset handling
├── utils/
│   ├── __init__.py
│   └── common.py         # Common utilities
├── config/
│   ├── mf_config.yaml
│   ├── mf_sweep_config.yaml
│   ├── ncf_config.yaml
│   ├── ncf_sweep_config.yaml
│   └── ncf_cold_start_config.yaml
├── __init__.py
├── train.py              # Unified training script
└── sweep.py              # Unified sweep script
```

## Key Components

### Models

1. **Base Model** (`models/base.py`):
   - `BaseRecommenderModel`: Abstract base class for all recommender models
   - `BiasedEmbeddingModel`: Base class for models with user/item embeddings and biases

2. **MF Models** (`models/mf.py`):
   - `MFModel`: Matrix Factorization model

3. **NCF Models** (`models/ncf.py`):
   - `NCFModel`: Neural Collaborative Filtering model

### Trainers

1. **Base Trainer** (`trainers/base.py`):
   - `BaseTrainer`: Base trainer class with common training logic

2. **Weighted Trainer** (`trainers/weighted.py`):
   - `WeightedTrainer`: Trainer that applies weights to training examples

3. **NCF Trainer** (`trainers/ncf.py`):
   - `NCFTrainer`: Trainer specialized for NCF models

### Datasets

1. **Base Dataset** (`datasets/base.py`):
   - `AmazonDataset`: Base dataset class

2. **Dataset Factory** (`datasets/factory.py`):
   - `DatasetFactory`: Factory for creating datasets

3. **Cold Start Dataset** (`datasets/cold_start.py`):
   - `TimeBasedColdStartDatasetFactory`: Factory for creating cold start datasets

### Training Scripts

1. **Unified Training Script** (`train.py`):
   - Command-line interface for training any model
   - Configurable via YAML files or command-line arguments
   - Supports all model types and training scenarios

2. **Unified Sweep Script** (`sweep.py`):
   - Command-line interface for running sweep experiments
   - Configurable via YAML files
   - Supports all model types and training scenarios

## Implementation Plan

1. Create the new directory structure
2. Move and refactor model implementations
3. Move and refactor trainer implementations
4. Move and refactor dataset implementations
5. Create unified training and sweep scripts
6. Update configuration files
7. Test all models and training scenarios