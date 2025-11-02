import logging
import os
from pymongo import MongoClient, errors
import pandas as pd
from dotenv import load_dotenv

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --- Load Environment Variables ---
load_dotenv()
MONGO_DB_URI = os.getenv("MONGO_DB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

def connect_to_db():
    """Establish MongoDB connection with error handling."""
    try:
        client = MongoClient(MONGO_DB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Check connection
        logging.info("✅ Connected to MongoDB successfully.")
        return client
    except errors.ServerSelectionTimeoutError as e:
        logging.error(f"❌ Could not connect to MongoDB: {e}")
        raise SystemExit("Exiting — MongoDB connection failed.")

def load_data():
    """Load user interactions, shows, and episodes from MongoDB."""
    client = connect_to_db()
    db = client[MONGO_DB_NAME]

    try:
        interactions = list(db.user_interactions.find())
        shows = list(db.shows.find())
        # For movies, find episodes with type: "individual"
        episodes = list(db.episodes.find({"type": "individual"}))
        logging.info(f"📊 Loaded {len(interactions)} interactions, {len(shows)} shows, and {len(episodes)} episodes (movies).")
    except Exception as e:
        logging.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        client.close()
        logging.info("🔒 MongoDB connection closed.")

    df_interactions = pd.DataFrame(interactions)
    df_shows = pd.DataFrame(shows)
    df_episodes = pd.DataFrame(episodes)
    return df_interactions, df_shows, df_episodes


if __name__ == "__main__":
    df_interactions, df_shows, df_episodes = load_data()
    logging.info(f"Interactions shape: {df_interactions.shape}, Shows shape: {df_shows.shape}, Episodes shape: {df_episodes.shape}")
