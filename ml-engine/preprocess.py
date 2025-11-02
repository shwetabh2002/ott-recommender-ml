import pandas as pd
import pytz

def preprocess_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps interactions to numeric rating and converts timestamp to Indian timezone.
    Uses ratingEquivalent if available, otherwise maps interaction types to ratings.
    """
    # Map interaction types to ratings (weights)
    interaction_weights = {
        "superlike": 1.5,
        "like": 1.0,
        "watchlist": 0.9,
        "play": 0.8,
        "playStart": 0.7,
        "download": 0.6,
        "click": 0.5,  # thumbnail_click
        "dislike": 0.0,
    }
    
    # Convert timestamp to Indian timezone if timestamp column exists
    if "timestamp" in df.columns:
        try:
            # Convert timestamp to datetime if it's a string
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Convert to Indian timezone (IST)
            ist = pytz.timezone("Asia/Kolkata")
            df["timestamp"] = df["timestamp"].dt.tz_convert(ist)
        except Exception as e:
            print(f"Warning: Could not convert timestamps: {e}")
    
    # Use ratingEquivalent if available, otherwise map interaction types
    if "ratingEquivalent" in df.columns:
        # Use ratingEquivalent, but fill missing values with interaction mapping
        df["rating"] = df["ratingEquivalent"].fillna(
            df["interaction"].map(interaction_weights)
        )
    else:
        # Map interaction types to ratings
        df["rating"] = df["interaction"].map(interaction_weights).fillna(0.5)
    
    # Return userId and contentSlug (instead of assetId)
    if "contentSlug" not in df.columns:
        raise ValueError("contentSlug column not found in interactions data")
    
    return df[["userId", "contentSlug", "rating"]]


if __name__ == "__main__":
    import data_loader
    df, _, _ = data_loader.load_data()
    print(preprocess_interactions(df).head())
