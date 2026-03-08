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

# 🛠️ UPDATE: More robust CORS for Vercel
# Adding both the production domain and the specific preview domain causing issues
origins = [
    "https://dlsca-frontend-3jgl.vercel.app",
    "https://dlsca-frontend-3jgl-kdcmm08q1-fahads-projects-4ecec35f.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Or use ["*"] temporarily to absolutely rule out CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# 🌐 PRODUCTION URLS
# We use the main production URL for the JWKS keys
JWKS_URL = "https://dlsca-frontend-3jgl.vercel.app/api/auth/jwks"
jwks_client = PyJWKClient(JWKS_URL)

def verify_user_token(token: str):
    try:
        unverified_header = jwt.get_unverified_header(token)
        token_alg = unverified_header.get("alg")
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # 🛠️ UPDATE: If you get a 403, it's often an audience mismatch.
        # This MUST match the 'baseURL' set in your SvelteKit Better Auth config exactly.
        return jwt.decode(
            token, 
            signing_key.key, 
            algorithms=[token_alg], 
            audience="https://dlsca-frontend-3jgl.vercel.app"
        )
    except Exception as e:
        print(f"❌ JWT Verification Error: {str(e)}")
        # Adding more detail to the error so you can see it in Railway logs
        raise HTTPException(status_code=401, detail=f"Unauthorized: {str(e)}")

def downsample_trace(arr: np.ndarray, max_points: int = 1000) -> list:
    if len(arr) <= max_points:
        return arr.tolist()
    
    chunk_size = len(arr) // (max_points // 2)
    trimmed_len = chunk_size * (max_points // 2)
    arr_trimmed = arr[:trimmed_len]
    
    reshaped = arr_trimmed.reshape(-1, chunk_size)
    min_vals = reshaped.min(axis=1)
    max_vals = reshaped.max(axis=1)
    
    compressed = np.empty(min_vals.size + max_vals.size, dtype=arr.dtype)
    compressed[0::2] = min_vals
    compressed[1::2] = max_vals
    
    return compressed.tolist()

@app.get("/api/secure-data")
def get_secure_data(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = verify_user_token(credentials.credentials)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    file_path = os.path.join(data_dir, "leakage_results.npy")
    
    if not os.path.exists(file_path):
        return {
            "status": "authorized",
            "message": "Connected to Railway! No data found yet.",
            "data": { "results": [], "filename": "None" }
        }

    trace_array = np.load(file_path)
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
        compressed_trace = downsample_trace(trace_array)
        
        return {
            "status": "authorized",
            "user_id": payload.get("sub"),
            "data": { "results": compressed_trace, "filename": file.filename }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid .npy file format")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Using 0.0.0.0 is mandatory for Railway
    uvicorn.run(app, host="0.0.0.0", port=port)