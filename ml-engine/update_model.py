import joblib
from datetime import datetime
from data_loader import load_data
from preprocess import preprocess_interactions
from train_model import train_model

if __name__ == "__main__":
    df_interactions, df_shows, df_episodes = load_data()
    df_interactions = preprocess_interactions(df_interactions)
    model, matrix, sparse, metadata = train_model(df_interactions, df_shows, df_episodes)
    joblib.dump(model, f"model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
    print("✅ Model updated and saved")
    print(f"   Content metadata: {len(metadata)} items")
