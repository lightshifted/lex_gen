from fastapi import FastAPI, Request
from typing import Dict
from http import HTTPStatus
from datetime import datetime
from functools import wraps
from config import config 
from config.config import logger
from pathlib import Path
from operations.main import generate_text

def construct_response(f):
    """Construct a JSON response for an endpoint"""
    
    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method" : request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response
    return wrap

# Application
app = FastAPI(
    title="LexGen",
    description="Philosophical text generator",
    version=0.1,
)


# Endpoint
@app.get("/")
@construct_response
def _index(request: Request) -> Dict:
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {}
    }
    return response


