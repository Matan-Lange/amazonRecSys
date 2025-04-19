# Amazon Recommendation System

A recommendation system that leverages multiple data sources to predict user preferences for Amazon products.

## Project Overview

This project implements recommendation models for predicting user-item ratings in both warm-item and cold-item scenarios. The system uses collaborative filtering techniques, including Matrix Factorization (MF) and Neural Collaborative Filtering (NCF) with image embeddings.

### Rating Prediction Tasks

- **Warm Items**: Items present in the training dataset
- **Cold Items**: Items with no prior interactions but available metadata and images

### Models

- **Matrix Factorization (MF)**: Traditional collaborative filtering with optional biases
- **Neural Collaborative Filtering (NCF)**: Deep learning approach that incorporates image embeddings

## Project Structure


### Warm Item-based Baselines

| Model         | RMSE   |
|--------------|--------|
| Global Mean  | 1.2024 |
| User Bias    | 1.1728 |
| Item Bias    | 1.1796 |
| User+Item Bias| 1.1584 |

## Usage Examples

### Running Matrix Factorization (MF) Model

#### Basic MF for Warm Items
```bash
python -m rating.train --model_type mf --trainer_type base --scenario warm --embed_dim  8 --batch_size 1024 --learning_rate 0.005 --epochs 4
```

#### MF with Weighted Training for Warm Items
```bash
python -m rating.train --model_type mf --trainer_type weighted --scenario warm --embed_dim  8 --batch_size 1024 --learning_rate 0.005 --epochs 4
```

#### MF with Configuration File
```bash
python -m rating.train --config rating\config\mf_config.yaml
```

### Running Neural Collaborative Filtering (NCF) Model

#### Basic NCF for Warm Items
```bash
python -m rating.train --model_type ncf --trainer_type ncf --scenario warm --batch_size 1024 --learning_rate 0.001 --epochs 4
```

#### NCF with Image Embeddings for Warm Items
```bash
python -m rating.train --model_type ncf --trainer_type ncf --scenario warm --image_model fashion_clip --batch_size 1024 --learning_rate 0.001 --epochs 4
```

#### NCF for Cold Start Items
```bash
python -m rating.train --model_type ncf --trainer_type ncf --scenario cold_start --image_model fashion_clip --batch_size 1024 --learning_rate 0.001 --epochs 4
```

#### NCF with Configuration File
```bash
python -m rating.train --config rating\config\ncf_config.yaml
```

### Running Hyperparameter Sweeps

#### MF Sweep
```bash
python -m rating.sweep --config rating\config\mf_sweep_config.yaml
```

#### NCF Sweep
```bash
python -m rating.sweep --config rating\config\ncf_sweep_config.yaml
```

#### NCF Cold Start Sweep
```bash
python -m rating.sweep --config rating\config\ncf_cold_start_config.yaml
```

