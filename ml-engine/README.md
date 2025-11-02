# ML Engine - OTT Recommendation System

This directory contains the core machine learning engine for the OTT recommendation system.

## 📁 File Structure

| File | Purpose |
|------|---------|
| `app.py` | FastAPI application and API endpoints |
| `data_loader.py` | MongoDB connection and data loading |
| `preprocess.py` | Data preprocessing and rating mapping |
| `train_model.py` | ALS model training with metadata extraction |
| `recommend.py` | Hybrid recommendation generation |
| `content_based.py` | 6D content-based scoring (enhanced) |
| `baseline_content_based.py` | Simple content-based scoring (baseline) |
| `evaluation.py` | Evaluation metrics (Precision, Recall, NDCG, Coverage) |
| `test_evaluation.py` | Full evaluation pipeline |
| `seed_data.py` | Sample data generator for testing |
| `update_model.py` | Model retraining script |

---

## 🚀 Quick Start

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create `.env` file:
```env
MONGO_DB_URI=mongodb://localhost:27017
MONGO_DB_NAME=ott_db
PORT=8000
```

### 3. Run

```bash
uvicorn app:app --reload --port 8000
```

---

## 📖 Module Documentation

### `app.py`
**FastAPI Application**
- Startup: Loads data, trains model
- Endpoint: `GET /recommendations/{user_id}`
- Returns: List of recommended content slugs

**Key Functions:**
- `startup_event()`: Loads data and trains model
- `get_recommendations(user_id)`: Main API endpoint

### `data_loader.py`
**MongoDB Data Loading**
- Connects to MongoDB
- Loads: `user_interactions`, `shows`, `episodes`
- Returns: Three DataFrames

**Key Functions:**
- `connect_to_db()`: MongoDB connection
- `load_data()`: Load all collections

### `preprocess.py`
**Data Preprocessing**
- Maps interaction types to numeric ratings
- Converts timestamps to IST
- Returns: `userId`, `contentSlug`, `rating`

**Interaction Weights:**
- `superlike`: 1.5
- `like`: 1.0
- `watchlist`: 0.9
- `play`: 0.8
- `playStart`: 0.7
- `download`: 0.6
- `click`: 0.5
- `dislike`: 0.0

### `train_model.py`
**Model Training**
- Extracts metadata from shows/episodes
- Creates user-item matrix (includes ALL content)
- Trains ALS collaborative filtering model
- Returns: model, user_item_matrix, sparse_matrix, content_metadata

**Extracted Metadata:**
- Genres, Sub-genres
- Artists
- Type, Format
- Moods, Themes
- Premium status
- Compliance ratings

### `recommend.py`
**Recommendation Generation**
- Hybrid approach: 70% Collaborative + 30% Content-Based
- Uses native `model.recommend()` with proper matrix slicing
- Filters already-liked items
- Returns: Top N content slugs

**Key Functions:**
- `recommend_for_user()`: Main recommendation function

### `content_based.py`
**Enhanced Content-Based Scoring (6D Model)**
- Multi-dimensional analysis
- Feature weights:
  - Genres: 40%
  - Artists: 20%
  - Type & Format: 15%
  - Moods: 10%
  - Themes: 10%
  - Premium: 5%

**Key Functions:**
- `compute_content_based_scores()`: Computes scores for all items

### `evaluation.py`
**Evaluation Framework**
- Precision@K, Recall@K, NDCG@K, Coverage
- Baseline vs Enhanced comparison
- Train/test splitting
- Ground truth preparation

**Key Functions:**
- `precision_at_k()`, `recall_at_k()`, `ndcg_at_k()`, `coverage()`
- `evaluate_recommendations()`, `compare_models()`
- `split_interactions()`, `prepare_ground_truth()`

---

## 🔧 Usage Examples

### Train Model

```python
from data_loader import load_data
from preprocess import preprocess_interactions
from train_model import train_model

df_interactions, df_shows, df_episodes = load_data()
df_interactions = preprocess_interactions(df_interactions)
model, matrix, sparse, metadata = train_model(df_interactions, df_shows, df_episodes)
```

### Generate Recommendations

```python
from recommend import recommend_for_user

recommendations = recommend_for_user(
    model, matrix, sparse, "user_1",
    content_metadata=metadata, top_n=10
)
```

### Run Evaluation

```python
python test_evaluation.py
```

---

## 🧪 Testing

### Generate Sample Data

```bash
python seed_data.py
```

### Test Individual Components

```bash
python data_loader.py
python preprocess.py
python train_model.py
python recommend.py
```

---

## 📊 Dependencies

See `requirements.txt`:
- `pandas` - Data manipulation
- `pymongo` - MongoDB connection
- `implicit` - ALS collaborative filtering
- `scipy` - Sparse matrices
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `joblib` - Model serialization
- `python-dotenv` - Environment variables
- `pytz` - Timezone handling
- `numpy` - Numerical operations

---

## 🐛 Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Model State

```python
print(f"Users: {user_item_matrix.shape[0]}")
print(f"Items: {user_item_matrix.shape[1]}")
print(f"Content metadata: {len(content_metadata)}")
```

---

## 📝 Notes

- **Matrix Format**: Must be CSR and float64 for implicit library
- **User-Item Matrix**: Includes ALL content (even with 0 interactions)
- **Content Metadata**: Extracted at training time, used for recommendations
- **Cold Start**: Works with content-based only if no interactions

---

## 🔗 Related Documentation

- [Main README](../README.md)
- [Hybrid Recommendation Explained](../HYBRID_RECOMMENDATION_EXPLAINED.md)
- [Evaluation Framework](../EVALUATION_FRAMEWORK.md)
- [Enhanced Metadata](../ENHANCED_METADATA_EXPLAINED.md)

