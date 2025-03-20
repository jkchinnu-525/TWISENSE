from fastapi import FastAPI
from .main import app
import uvicorn

def create_app() -> FastAPI:
    return app

if __name__ == "__main__":
    uvicorn.run("backend:create_app()", host='0.0.0.0', port=8000, reload=True)