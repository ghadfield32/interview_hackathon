# Create logs dir early when package is imported by Uvicorn workers
import os
os.makedirs("logs", exist_ok=True) 
