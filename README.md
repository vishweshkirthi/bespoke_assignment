# FastText Document Quality Classifier

A FastText-based web service for classifying document quality with real-time training progress and autotune support.

## Quick Start

### Option 1: Docker (Recommended)

1. **Run with Docker Compose**:
   ```bash
   cd docker
   docker-compose up --build
   ```

2. **Access the UI**: Open `http://localhost:8000/ui`

### Option 2: Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   If you want to use the same sample of data that I used from [news dataset](https://huggingface.co/datasets/stanford-oval/ccnews/tree/main)  and [common crawl](https://huggingface.co/datasets/agentlans/common-crawl-sample/tree/main/en)
   Just run:
   ```bash
   cd scripts
   python create_datasets.py
   ```

3. **Start the API**:
   ```bash
   python app.py
   ```

4. **Access the UI**: Open `http://localhost:8000/ui`

## Project Structure

```
bespoke_assignment/
├── app.py                      # Main FastAPI application
├── src/                        # Source code modules
│   ├── preprocessing.py        # Text preprocessing
├── scripts/                    # Utility scripts
│   ├── create_datasets.py      # Dataset creation
│   ├── train_fasttext.py       # Standalone training
│   └── predict_fasttext.py     # Standalone prediction
├── data/                       # Data files
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed training data
│   └── sample/                # Small sample files
├── models/                     # Trained models
├── docs/                      # Documentation
├── tests/                     # Test files
├── docker/                    # Docker configuration
│   ├── Dockerfile             # Container definition
│   ├── docker-compose.yml     # Orchestration
│   └── README.md             # Docker usage guide
└── .dockerignore              # Docker build exclusions
```

## Usage

### Web Interface
- Upload training files with positive examples
- Optionally enable FastText autotune with validation data
- Score individual texts for quality
- Monitor training progress in real-time

### API Endpoints
- `GET /` - Health check
- `POST /train` - Train model with file upload
- `POST /score` - Score text quality
- `GET /model/status` - Check model status
- `WS /ws` - Training progress WebSocket

### Command Line Scripts
```bash
# Create datasets from raw data
python scripts/create_datasets.py

# Train model standalone
python scripts/train_fasttext.py --positive data/processed/train_positive.txt

# Score texts
python scripts/predict_fasttext.py --input data/processed/combined_test.txt
```

## Data Format

FastText format with labels:
```
__label__high This is a high quality document with proper structure and content.
__label__low bad doc very short no useful information
```

## Text Requirements

- Minimum 50 characters
- Minimum 10 words

## Documentation

See `docs/api_usage.md` for detailed API documentation.
