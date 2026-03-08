from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import jwt
from jwt import PyJWKClient
import uvicorn
import numpy as np
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
JWKS_URL = "http://localhost:5173/api/auth/jwks"
jwks_client = PyJWKClient(JWKS_URL)

def verify_user_token(token: str):
    try:
        unverified_header = jwt.get_unverified_header(token)
        token_alg = unverified_header.get("alg")
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(token, signing_key.key, algorithms=[token_alg], audience="http://localhost:5173")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Unauthorized: {str(e)}")

# ✨ NEW: Min-Max Downsampling Function ✨
def downsample_trace(arr: np.ndarray, max_points: int = 1000) -> list:
    """Reduces array size while preserving leakage spikes for the UI"""
    if len(arr) <= max_points:
        return arr.tolist()
    
    # Calculate how many points go into each chunk
    chunk_size = len(arr) // (max_points // 2)
    
    # Trim excess data that doesn't fit evenly into chunks
    trimmed_len = chunk_size * (max_points // 2)
    arr_trimmed = arr[:trimmed_len]
    
    # Reshape and grab the minimum and maximum values of each chunk
    reshaped = arr_trimmed.reshape(-1, chunk_size)
    min_vals = reshaped.min(axis=1)
    max_vals = reshaped.max(axis=1)
    
    # Interleave the mins and maxes to create the final compressed trace
    compressed = np.empty(min_vals.size + max_vals.size, dtype=arr.dtype)
    compressed[0::2] = min_vals
    compressed[1::2] = max_vals
    
    return compressed.tolist()

@app.get("/api/secure-data")
def get_secure_data(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = verify_user_token(credentials.credentials)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "leakage_results.npy")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="No trace data found on server.")

    trace_array = np.load(file_path)
    # Apply downsampling before sending!
    compressed_trace = downsample_trace(trace_array)
    
    return {
        "status": "authorized",
        "user_id": payload.get("sub"),
        "data": { "results": compressed_trace, "filename": "leakage_results.npy" }
    }

@app.post("/api/upload")
async def upload_trace(
    file: UploadFile = File(...), 
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    payload = verify_user_token(credentials.credentials)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        trace_array = np.load(file_path)
        # Apply downsampling before sending!
        compressed_trace = downsample_trace(trace_array)
        
        return {
            "status": "authorized",
            "user_id": payload.get("sub"),
            "data": { "results": compressed_trace, "filename": file.filename }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid .npy file format")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)