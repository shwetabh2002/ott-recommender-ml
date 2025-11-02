from fastapi import FastAPI, HTTPException
import logging
import os
import pandas as pd
from data_loader import load_data
from preprocess import preprocess_interactions
from train_model import train_model
from recommend import recommend_for_user
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="OTT Recommendation Engine")

model = None
user_item_matrix = None
sparse_matrix = None
content_metadata = None

# Get port from environment variable, default to 8000
PORT = int(os.getenv("PORT", "8000"))


@app.on_event("startup")
async def startup_event():
    global model, user_item_matrix, sparse_matrix, content_metadata
    try:
        logging.info(f"🚀 Starting Recommendation Engine server on port {PORT}...")
        df_interactions, df_shows, df_episodes = load_data()

        # Note: We allow empty interactions for hybrid approach (can use content-based only)
        df_interactions_processed = pd.DataFrame()
        if not df_interactions.empty:
            df_interactions_processed = preprocess_interactions(df_interactions)
        else:
            logging.warning("⚠️ No interactions found. Will use content-based recommendations only.")

        model, user_item_matrix, sparse_matrix, content_metadata = train_model(
            df_interactions_processed, 
            df_shows, 
            df_episodes
        )
        logging.info(f"✅ Model trained and ready for hybrid recommendations on port {PORT}.")
        logging.info(f"   📦 Content metadata loaded: {len(content_metadata)} items")
    except Exception as e:
        logging.error(f"❌ Startup failed: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    logging.info("🛑 Shutting down Recommendation Engine server...")


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    try:
        if model is None or user_item_matrix is None or sparse_matrix is None or content_metadata is None:
            raise HTTPException(status_code=500, detail="Model or metadata not loaded yet.")

        recommendations = recommend_for_user(
            model, 
            user_item_matrix, 
            sparse_matrix, 
            user_id,
            content_metadata=content_metadata
        )
        if not recommendations:
            logging.warning(f"No recommendations found for user {user_id}.")
            return {"user_id": user_id, "recommendations": []}

        logging.info(f"🎯 Generated {len(recommendations)} hybrid recommendations for user {user_id}.")
        return {"user_id": user_id, "recommendations": recommendations}

    except Exception as e:
        logging.error(f"❌ Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
