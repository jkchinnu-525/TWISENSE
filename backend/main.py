from fastapi import FastAPI,HTTPException,status
import schemas
from transformers import pipeline
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import tweepy # type: ignore
import os

# Loads environment variables
load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://twisense.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initializes Twitter client
client = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
)

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

last_api_call = {}

@app.options("/predict")
async def options_predict():
    return {}

@app.post("/predict")
async def predict_sentiment(request: schemas.TextRequest):
    try:
        result = classifier(request.text)
        return {
            "sentiment": result[0]['label'],
            "confidence": result[0]['score']
        }
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    
@app.get("/tweets")
async def get_tweets(query: str = "Bitcoin", max_results: int = 10):
    global last_api_call
    current_time = datetime.now()
    if query in last_api_call:
        time_since_last_call = current_time - last_api_call[query]
        if time_since_last_call < timedelta(minutes=15):
            wait_time = 15 - (time_since_last_call.total_seconds() / 60)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit for this query. Please try again in {wait_time:.1f} minutes.",
                headers={"Retry-After": str(int(wait_time * 60))}  # Seconds to wait
            )
    try:
        tweets = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at","text","author_id"],
            user_fields=["username","id"],
            expansions=["author_id"]
            )
        if not hasattr(tweets, 'data') or not tweets.data:
            return []
        print(tweets)
        users = {user.id: user for user in tweets.includes['users']}
        tweets_list = [{
            "text":tweet.text,
             "created_at":tweet.created_at,            
             "user_id": tweet.author_id,
             "username":users.get(tweet.author_id,{}).get('username','Unknown')
             } for tweet in tweets.data]
        last_api_call[query] = current_time
        return tweets_list
    except tweepy.errors.TooManyRequests:
        last_api_call[query] = current_time
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Twitter API Rate limit exceeded. Please try again after 15min.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    