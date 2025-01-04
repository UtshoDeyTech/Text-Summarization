from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import openai
from app.service.log_client import logger
from config import OPENAI_API_KEY

router = APIRouter()
openai.api_key = OPENAI_API_KEY


