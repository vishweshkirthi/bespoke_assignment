# API Usage Guide

## FastText Document Quality Classifier API

### Starting the Server

```bash
cd bespoke_assignment
pip install -r requirements.txt
python app.py
```

The API will be available at `http://localhost:8000`

### Web UI

Access the interactive web interface at `http://localhost:8000/ui`

### API Endpoints

#### 1. Health Check
```
GET /
```
Returns API status and available endpoints.

#### 2. Model Status
```
GET /model/status
```
Check if a model is currently loaded.

#### 3. Train Model
```
POST /train
```
Upload training data and optionally use FastText autotune.

**Form Data:**
- `file`: Training file with positive examples (format: `__label__high [text]`)
- `validation_file` (optional): Validation file for autotune (should ideally contain both `__label__high [text]` and `__label__low [text]`)

**Response:**
```json
{
  "message": "Training started in background",
  "training_examples": 26000,
  "model_saved": "models/document_quality_model.bin",
  "autotune_used": true
}
```

#### 4. Score Text
```
POST /score
```
Score a text for quality.

**Request Body:**
```json
{
  "text": "Your document text to classify..."
}
```

**Response:**
```json
{
  "text": "Your document text...",
  "predicted_label": "high",
  "confidence": 0.9234,
  "is_high_quality": true
}
```

#### 5. WebSocket Training Progress
```
WS /ws
```
Real-time training progress updates.

### Text Requirements

All text inputs must meet these criteria:
- Minimum 50 characters
- Maximum 2000 characters  
- Minimum 10 words

### Data Preprocessing

All text is automatically preprocessed:
- Unicode normalization
- Lowercase conversion
- Punctuation standardization
- Whitespace normalization
- Control character removal

### Example Usage

```python
import requests

# Train model
with open('positive_examples.txt', 'rb') as f:
    response = requests.post('http://localhost:8000/train', files={'file': f})

# Score text
response = requests.post('http://localhost:8000/score', 
                        json={'text': 'This is a high-quality document with proper structure and content.'})
print(response.json())
```