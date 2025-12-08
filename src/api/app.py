"""
FastAPI application for term deposit prediction.

This API provides a /predict endpoint that takes raw customer data
and returns a prediction on whether they will subscribe to a term deposit.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
from pathlib import Path
import sys


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))