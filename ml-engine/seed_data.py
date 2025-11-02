from pymongo import MongoClient
import random
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_DB_URI = os.getenv("MONGO_DB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if not MONGO_DB_URI or not MONGO_DB_NAME:
    raise ValueError("MONGO_DB_URI and MONGO_DB_NAME must be set in .env file")

client = MongoClient(MONGO_DB_URI)
db = client[MONGO_DB_NAME]

# Get shows and episodes (movies) - shows have slug and format, episodes have slug and type
# Note: shows table doesn't have "type", episodes table doesn't have "format"
shows = list(db.shows.find({"status": "active","displayLanguage":"en"}, {"slug": 1, "format": 1}))
episodes = list(db.episodes.find({"status": "active", "type": "individual","displayLanguage":"en"}, {"slug": 1, "type": 1}))

# Combine shows and episodes for content slugs
all_content = []
for show in shows:
    if "slug" in show:
        all_content.append({
            "contentSlug": show["slug"],  # Map slug to contentSlug for interactions
            "type": "show",  # Shows don't have type field, default to "show"
            "format": show.get("format", "standard")  # Get format from show if available
        })

for episode in episodes:
    if "slug" in episode:
        all_content.append({
            "contentSlug": episode["slug"],  # Map slug to contentSlug for interactions
            "type": "movie",  # Episodes with type "individual" are movies
            "format": "standard"  # Episodes don't have format field, default to "standard"
        })

if not all_content:
    print("❌ No shows or episodes found with slug. Please check your database.")
    client.close()
    exit(1)

# Available interaction types
INTERACTION_TYPES = [
    "dislike",
    "download",
    "like",
    "play",
    "playStart",
    "superlike",
    "click",  # thumbnail_click
    "watchlist"
]

# Generate mock interactions with new schema
interactions = []
ist = pytz.timezone("Asia/Kolkata")

# Generate interactions for different users, profiles, and devices
user_ids = [f"user_{i}" for i in range(1, 6)]
profile_ids = [f"profile_{i}" for i in range(1, 4)]
device_ids = [f"device_{i}" for i in range(1, 4)]

for i in range(50):  # 50 interactions
    content = random.choice(all_content)
    
    # Rating equivalent mapping (optional, can be used if available)
    interaction_type = random.choice(INTERACTION_TYPES)
    rating_equiv_map = {
        "superlike": 1.5,
        "like": 1.0,
        "watchlist": 0.9,
        "play": 0.8,
        "playStart": 0.7,
        "download": 0.6,
        "click": 0.5,
        "dislike": 0.0,
    }
    rating_equivalent = rating_equiv_map.get(interaction_type, 0.5)
    
    # Generate timestamp in IST
    timestamp = datetime.now(ist)
    
    interactions.append({
        "contentSlug": content["contentSlug"],
        "type": content["type"],
        "format": content["format"],
        "interaction": interaction_type,
        "profileId": random.choice(profile_ids),
        "userId": random.choice(user_ids),
        "deviceId": random.choice(device_ids),
        "timestamp": timestamp.isoformat(),
        "ratingEquivalent": rating_equivalent
    })

if interactions:
    db.user_interactions.insert_many(interactions)
    print(f"✅ Inserted {len(interactions)} mock interactions with new schema.")
    print(f"   - Users: {len(user_ids)}")
    print(f"   - Profiles: {len(profile_ids)}")
    print(f"   - Devices: {len(device_ids)}")
    print(f"   - Content items: {len(all_content)}")
else:
    print("❌ No interactions generated.")

client.close()
