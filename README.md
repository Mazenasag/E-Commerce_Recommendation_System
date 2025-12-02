<div align="center">

# 🎯 E-Commerce Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A production-ready hybrid recommendation system that combines collaborative filtering (ALS) and content-based filtering (TF-IDF + SVD) to provide product-to-user recommendations for e-commerce platforms.**

---

### 🚀 Quick Start with Docker Hub

```bash
# Pull the pre-built image from Docker Hub
docker pull mazenasag/ecommerce-recommender:latest

# Run the API server with volume mounts
docker pull mazenasag/ecommerce-recommender:latest
docker run -d -p 8000:8000 mazenasag/ecommerce-recommender:latest



# Access the API
# - API Docs: http://localhost:8000/docs
# - Web UI: http://localhost:8000/static/index.html
```

**📦 Docker Image:** `mazenasag/ecommerce-recommender:latest`

---

</div>

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Data Description](#data-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Running with DVC](#-running-with-dvc-data-version-control)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Using Conda (recommended)
conda create -n recommender python=3.11 -y
conda activate recommender
pip install -r requirements.txt

# Or using pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

Place your CSV file in `data/raw/` with columns: `customer_id`, `product_id`, `product_name`, `event`, `event_date`

### 3. Train Model

```bash
# Option A: Direct Python script
python run_pipeline.py

# Option B: Using DVC (recommended for reproducibility)
dvc repro
```

### 4. Start API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Web UI**: http://localhost:8000/static/index.html

### 6. Docker (Alternative)

```bash
# Pull and run pre-built image
docker pull mazenasag/ecommerce-recommender:latest
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  mazenasag/ecommerce-recommender:latest

# Or use docker-compose
docker-compose up -d
```

## 🎯 Overview

This system provides intelligent product recommendations by analyzing user interactions (views, purchases, cart additions, ratings) and product content (names, descriptions). It uses a hybrid approach combining:

- **Collaborative Filtering (ALS)**: Learns user preferences from interaction patterns
- **Content-Based Filtering**: Uses product text features (TF-IDF + SVD) for similarity
- **Hybrid Method**: Combines both approaches for optimal recommendations

The system is designed for Arabic and English e-commerce platforms with support for bilingual product names.

## ✨ Features

- **Multiple Recommendation Methods**:
  - Hybrid (ALS + Content-Based) - Recommended
  - Collaborative Filtering (ALS)
  - Content-Based (TF-IDF + FAISS)
  - Popularity-Based (fallback)

- **Production-Ready Pipeline**:
  - Modular, maintainable code structure
  - End-to-end training pipeline
  - FastAPI REST API with OpenAPI docs
  - Comprehensive logging and error handling
  - Data version control (DVC) support

- **Performance Optimizations**:
  - FAISS for fast similarity search
  - Sparse matrix operations
  - Caching mechanisms
  - Async API support

- **Developer Experience**:
  - YAML-based configuration
  - Unit tests
  - Docker containerization
  - Web interface for testing

## 🏗️ Architecture

```
┌─────────────────┐
│   Raw Data      │
│   (CSV)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │
│ & Validation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Transform  │
│ - Cleaning      │
│ - Text Process  │
│ - ID Mapping    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│ - TF-IDF + SVD  │
│ - ALS Model     │
│ - FAISS Index   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Artifacts     │
│   (Saved)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI API    │
│  (Inference)    │
└─────────────────┘
```

## 📊 Data Description

### Input Data Format

The system expects a CSV file with the following columns:

- `customer_id`: Unique user identifier
- `product_id`: Unique product identifier
- `product_name`: Product name (Arabic/English)
- `event`: Interaction type (`purchased`, `cart`, `rating`, `wishlist`, `search_keyword`)
- `event_date`: Timestamp of the interaction

### Data Statistics

- **Users**: Label-encoded for efficient processing
- **Products**: Label-encoded with text embeddings
- **Interactions**: Weighted by event type and recency
- **User Segments**: Warm users (2+ interactions) vs Cold users (1 interaction)

### Event Weights

Different interaction types have different importance:
- `purchased`: 5.0 (highest)
- `cart`: 3.0
- `rating`: 2.5
- `wishlist`: 2.0
- `search_keyword`: 1.0

## 📁 Project Structure

```
E-Commerce_Recommendation_System/
│
├── config/                      # Configuration files
│   ├── config.yaml              # Main configuration (paths, hyperparameters)
│   └── logging.yaml             # Logging configuration
│
├── data/                        # Data directory
│   ├── raw/                     # Raw input data (CSV files)
│   ├── processed/               # Processed data (CSV)
│   └── external/                # External data sources
│
├── src/                         # Source code
│   ├── components/              # Core components
│   │   ├── data_ingestion.py   # Data loading and validation
│   │   ├── data_transformation.py  # Data cleaning and preprocessing
│   │   ├── model_trainer.py    # Model training (ALS, embeddings)
│   │   ├── model_evaluator.py  # Model evaluation
│   │   └── prediction.py       # Inference logic
│   ├── pipeline/                # End-to-end pipelines
│   │   ├── train_pipeline.py   # Training pipeline
│   │   └── predict_pipeline.py  # Prediction pipeline
│   ├── utils/                   # Utilities
│   │   ├── logger.py           # Logging utilities
│   │   ├── exception.py        # Custom exceptions
│   │   └── helpers.py          # Helper functions
│   └── config_loader.py         # Configuration loader
│
├── app/                         # FastAPI application
│   ├── main.py                  # FastAPI app and endpoints
│   ├── services/                # Business logic
│   │   └── recommender_service.py  # Recommendation service
│   ├── middleware.py            # Performance middleware
│   └── config.py                # API configuration
│
├── artifacts/                    # Saved models and artifacts
│   ├── als_model.pkl            # Trained ALS model
│   ├── faiss_index.bin           # FAISS similarity index
│   ├── product_embeddings.npy   # Product embeddings
│   ├── tfidf_vectorizer.pkl     # TF-IDF vectorizer
│   ├── svd_transformer.pkl       # SVD transformer
│   ├── id_mappings.json         # ID mappings
│   ├── train_matrix.npz         # Training matrix
│   ├── interaction_matrix.npz  # Full interaction matrix
│   ├── warm_user_info.json      # Warm user information
│   ├── product_to_users_lookup.json  # Product-user lookup
│   └── product_ids_list.json    # Product IDs list
│
├── static/                      # Web interface
│   ├── index.html               # Main UI
│   ├── user-history.html        # User history page
│   ├── script.js                # Frontend JavaScript
│   └── styles.css               # Styling
│
├── tests/                       # Unit tests
│   ├── test_data_ingestion.py
│   ├── test_data_transformation.py
│   └── test_model_trainer.py
│
├── EDA_and_notebooks_trails/    # Jupyter notebooks
│   ├── [eda.ipynb](EDA_and_notebooks_trails/eda.ipynb)                # Exploratory data analysis
│   ├── [data_preprocessing.ipynb](EDA_and_notebooks_trails/data_preprocessing.ipynb)  # Data preprocessing
│   └── [model_build.ipynb](EDA_and_notebooks_trails/model_build.ipynb)        # Model development
│
├── logs/                        # Application logs
├── reports/                     # Test reports
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── run_pipeline.py              # Training pipeline script
├── dvc.yaml                     # DVC pipeline configuration
├── Dockerfile                    # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- Docker (optional, for containerized deployment)

### Option 1: Conda Environment (Recommended)

```bash
# Create conda environment
conda create -n recommender python=3.10 -y
conda activate recommender

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi, faiss, implicit; print('✅ All packages installed')"
```

### Option 2: Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker (See Docker Deployment section)

## 💻 Usage

### 1. Prepare Data

Place your CSV file in `data/raw/` directory. The file should contain columns:
- `customer_id`, `product_id`, `product_name`, `event`, `event_date`

Update `config/config.yaml` with your data path if different.

### 2. Train the Model

```bash
# Run training pipeline
python run_pipeline.py

# Or with custom config
python run_pipeline.py --config config/config.yaml
```

This will:
- Load and preprocess data
- Create product embeddings (TF-IDF + SVD)
- Train ALS model
- Build FAISS index
- Save all artifacts to `artifacts/` directory

**Expected Output:**
```
Starting training pipeline...
Loading preprocessed data...
Train/Test split: 80,000 / 20,000
Creating product embeddings...
Products with embeddings: 15,000
Building FAISS index...
Training ALS model...
Building product-user lookup...
Saving artifacts...
Training pipeline complete!
```

### 3. Start the API Server

```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000/static/index.html

### 4. Make Recommendations

#### Using Python

```python
import requests

# Hybrid recommendations (recommended)
response = requests.post(
    "http://localhost:8000/recommendations",
    json={
        "product_id": 12345,
        "n": 10,
        "method": "hybrid"
    }
)
recommendations = response.json()
print(recommendations)
```

#### Using cURL

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 12345,
    "n": 10,
    "method": "hybrid"
  }'
```

#### Using Web Interface

1. Open http://localhost:8000/static/index.html
2. Enter a product ID
3. Select recommendation method
4. Click "Get Recommendations"

## 🔄 Running with DVC (Data Version Control)

DVC allows you to version control your data pipeline, track dependencies, and reproduce experiments. Here's how to use DVC with this project:

### Install DVC

```bash
# Using pip
pip install dvc

# Using conda
conda install -c conda-forge dvc

# Verify installation
dvc --version
```

### Initialize DVC (First Time Only)

```bash
# Navigate to project root
cd E-Commerce_Recommendation_System

# Initialize DVC repository
dvc init

# Optional: Configure remote storage (S3, GCS, Azure, etc.)
# dvc remote add -d myremote s3://my-bucket/dvc-storage
```

### Run the Pipeline with DVC

#### Option 1: Run Entire Pipeline

```bash
# Run all stages in the pipeline
dvc repro

# This will execute:
# 1. data_ingestion stage
# 2. data_transformation stage  
# 3. model_training stage
```

#### Option 2: Run Specific Stage

```bash
# Run only data transformation
dvc repro data_transformation

# Run only model training (requires processed data)
dvc repro model_training

# Run from a specific stage onwards
dvc repro model_training
```

#### Option 3: Force Re-run (Ignore Cache)

```bash
# Force re-run all stages
dvc repro --force

# Force re-run specific stage
dvc repro --force model_training
```

### View Pipeline Status

```bash
# Show pipeline graph
dvc dag

# Show pipeline status
dvc status

# Show detailed pipeline information
dvc pipeline show

# List all stages
dvc stage list
```

### Check Pipeline Dependencies

```bash
# Show what changed since last run
dvc status

# Show pipeline visualization
dvc dag --dot | dot -Tpng -o pipeline.png
```

### View Pipeline Outputs

After running `dvc repro`, check the outputs:

```bash
# Check processed data
ls -lh data/processed/

# Check artifacts
ls -lh artifacts/

# View specific artifact info
dvc metrics show artifacts/config.json
```

### Common DVC Commands

```bash
# Show pipeline stages and their status
dvc stage list

# Remove all outputs and cache
dvc destroy

# Show pipeline graph (text)
dvc dag

# Show pipeline graph (visual - requires graphviz)
dvc dag --dot | dot -Tpng -o pipeline.png

# Check if pipeline is up to date
dvc status

# Pull data from remote storage
dvc pull

# Push data to remote storage
dvc push

# Show differences between versions
dvc diff
```

### Testing the Pipeline

#### Step-by-Step Test

```bash
# 1. Ensure data is in place
ls data/raw/csv_for_case_study_V1.csv

# 2. Check DVC configuration
cat dvc.yaml

# 3. Run the pipeline
dvc repro

# 4. Verify outputs
dvc status

# 5. Check generated artifacts
ls artifacts/
```

#### Expected Output

When running `dvc repro`, you should see:

```
Running stage 'data_ingestion':
> python -c "from src.components.data_ingestion import DataIngestion; ..."
Updating lock file 'dvc.lock'

Running stage 'data_transformation':
> python -c "from src.components.data_transformation import DataTransformation; ..."
Updating lock file 'dvc.lock'

Running stage 'model_training':
> python run_pipeline.py --config config/config.yaml
Updating lock file 'dvc.lock'

Pipeline stages executed successfully!
```

### Troubleshooting DVC

#### Issue: "Stage is up to date"

If DVC says a stage is up to date but you want to re-run:

```bash
# Force re-run
dvc repro --force

# Or remove the stage output first
dvc remove <stage_name>.dvc
dvc repro
```

#### Issue: "Dependencies changed"

If you modified code but DVC doesn't detect it:

```bash
# Check what DVC thinks changed
dvc status

# Force re-run from a specific stage
dvc repro --force model_training
```

#### Issue: "Missing dependencies"

If a stage fails due to missing files:

```bash
# Check dependencies
dvc dag

# Ensure all dependencies exist
ls data/raw/
ls data/processed/
```

#### Clear DVC Cache

```bash
# Remove cache (keeps outputs)
dvc cache dir

# Clear specific cache
dvc gc --workspace
```

### DVC Best Practices

1. **Commit DVC files**: Always commit `dvc.yaml` and `dvc.lock` to Git
2. **Use remote storage**: Set up remote storage for large files
3. **Tag experiments**: Use Git tags with DVC to track experiments
4. **Check status**: Run `dvc status` before committing changes
5. **Document changes**: Update `dvc.yaml` when modifying pipeline

### Example Workflow

```bash
# 1. Make changes to code
vim src/components/model_trainer.py

# 2. Check what needs to be re-run
dvc status

# 3. Re-run affected stages
dvc repro model_training

# 4. Verify outputs
ls artifacts/

# 5. Commit changes
git add dvc.lock dvc.yaml
git commit -m "Updated model training parameters"
```

### DVC Pipeline Visualization

To visualize your pipeline:

```bash
# Install graphviz (if not installed)
# Windows: choco install graphviz
# Linux: sudo apt-get install graphviz
# Mac: brew install graphviz

# Generate pipeline graph
dvc dag --dot | dot -Tpng -o pipeline.png

# View the image
# Windows: start pipeline.png
# Linux: xdg-open pipeline.png
# Mac: open pipeline.png
```

## 📡 API Documentation

### Endpoints

#### 1. Get Recommendations

**POST** `/recommendations`

Get users who might be interested in a product.

**Request Body:**
```json
{
  "product_id": 12345,
  "n": 10,
  "method": "hybrid"
}
```

**Parameters:**
- `product_id` (int, required): Product ID to get recommendations for
- `n` (int, optional, default=10): Number of recommendations (1-100)
- `method` (str, optional, default="hybrid"): Method to use
  - `"hybrid"`: Combines ALS + Content-Based (recommended)
  - `"als"`: Collaborative filtering only
  - `"content"`: Content-based only
  - `"popularity"`: Most popular products

**Response:**
```json
{
  "product_id": 12345,
  "method": "hybrid",
  "recommendations": [
    {
      "user_id": 1001,
      "score": 0.85,
      "reason": "Similar users also interacted with this product"
    },
    ...
  ],
  "total_recommendations": 10
}
```

#### 2. Get Similar Products

**GET** `/similar-products/{product_id}`

Find products similar to a given product.

**Query Parameters:**
- `n` (int, optional, default=10): Number of similar products

**Response:**
```json
{
  "product_id": 12345,
  "similar_products": [
    {
      "product_id": 12346,
      "similarity": 0.92
    },
    ...
  ]
}
```

#### 3. Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "artifacts_loaded": true
}
```

### Interactive API Documentation

Visit http://localhost:8000/docs for Swagger UI with interactive testing.

## 🐳 Docker Deployment

### Option 1: Pull Pre-built Image from Docker Hub (Recommended)

```bash
# Pull the latest image
docker pull mazenasag/ecommerce-recommender:latest

# Run the API server
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  --name ecommerce-recommender \
  mazenasag/ecommerce-recommender:latest

# View logs
docker logs -f ecommerce-recommender

# Stop container
docker stop ecommerce-recommender
```

### Option 2: Generate Artifacts Inside Container

If you need to train the model inside the container:

```bash
# Pull image
docker pull mazenasag/ecommerce-recommender:latest

# Run training pipeline (mount data directory)
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  mazenasag/ecommerce-recommender:latest \
  python run_pipeline.py

# Then start the API
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  mazenasag/ecommerce-recommender:latest
```

### Option 3: Build from Source

```bash
# Build image locally
docker build -t ecommerce-recommender:latest .

# Or using docker-compose
docker-compose build
```

### Run with Docker

```bash
# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  ecommerce-recommender:latest

# Or using docker-compose
docker-compose up -d
```

### Docker Compose

The `docker-compose.yml` includes:
- API service on port 8000
- Volume mounts for data and artifacts
- Environment variables configuration

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ⚙️ Configuration

All configuration is in `config/config.yaml`:

### Model Parameters

```yaml
model:
  als:
    factors: 64              # Latent factors
    regularization: 0.3      # Regularization strength
    iterations: 30           # Training iterations
    alpha: 40                # Confidence scaling
  
  embeddings:
    max_features: 100        # TF-IDF max features
    ngram_range: [1, 2]      # N-gram range
    min_df: 3                # Minimum document frequency
    n_components: 50        # SVD components
```

### Event Weights

```yaml
event_weights:
  purchased: 5.0
  cart: 3.0
  rating: 2.5
  wishlist: 2.0
  search_keyword: 1.0
```

### API Configuration

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: true
```

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_model_trainer.py

# Run with coverage
python -m pytest tests/ --cov=src
```

### Test API Endpoints

```bash
# Using pytest (if test files exist)
pytest tests/test_api.py

# Manual testing via Swagger UI
# Visit http://localhost:8000/docs
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Artifacts Not Found

**Error**: `Artifacts directory 'artifacts' not found`

**Solution**: Run the training pipeline first:
```bash
python run_pipeline.py
```

#### 2. Import Errors

**Error**: `ModuleNotFoundError`

**Solution**: Ensure you're in the correct environment and dependencies are installed:
```bash
pip install -r requirements.txt
```

#### 3. FAISS Installation Issues

**Error**: `ImportError: cannot import name 'IndexFlatL2'`

**Solution**: Install FAISS:
```bash
# CPU version
pip install faiss-cpu

# GPU version (if available)
pip install faiss-gpu
```

#### 4. Memory Issues

**Error**: `MemoryError` during training

**Solution**: 
- Reduce `max_features` in `config/config.yaml`
- Reduce `n_components` for SVD
- Process data in batches

#### 5. Port Already in Use

**Error**: `Address already in use`

**Solution**: Change port in `config/config.yaml` or use:
```bash
uvicorn app.main:app --port 8001
```

### Logs

Check logs in `logs/` directory for detailed error information.

## 📈 Performance

### Expected Performance

- **Training Time**: 5-15 minutes (depending on data size)
- **API Latency**: 
  - Average: <500ms
  - P95: <1000ms
- **Throughput**: 50-100 requests/second

### Optimization Tips

1. **Increase Workers**: Set `workers: 4` in production
2. **Enable Caching**: Already enabled in `RecommenderService`
3. **Use GPU**: Install `faiss-gpu` for faster similarity search
4. **Batch Processing**: Process multiple recommendations in batches

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Mazen Asag**

