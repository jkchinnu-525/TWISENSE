from fastapi import FastAPI,HTTPException,status
import schemas
from transformers import pipeline
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import tweepy # type: ignore
import os

# Loads environment variables
load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initializes Twitter client
client = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
    # access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
    # access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
)

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.post("/predict")
async def predict_sentiment(request: schemas.TextRequest):
    try:
        result = classifier(request.text)
        return {
            "sentiment": result[0]['label'],
            "confidence": result[0]['score']
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    
@app.get("/tweets")
async def get_tweets(query: str = "Bitcoin", max_results: int = 10):
    try:
        tweets = client.search_recent_tweets(query=query,max_results=max_results,tweet_fields=["created_at","text","author_id"])
        tweets_list = [{"text":tweet.text, "created_at":tweet.created_at, "author_id": tweet.author_id} for tweet in tweets.data]
        return tweets_list
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
