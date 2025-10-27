#!/usr/bin/env python3
"""
Home GPU Server for RTX 3070 - WITH REAL-TIME STREAMING
Proven v7 algorithm + Server-Sent Events for live animation

FEATURES:
- /generate_stream: Real-time SSE updates during generation
- /generate: Legacy synchronous endpoint (same algorithm)
- Uses the proven v7 alternating search algorithm
- Progress updates every 50 lines
- New line events for real-time drawing

Run: python home_gpu_server.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
import logging
import base64
import io
import json
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Home GPU Server - RTX 3070 with Streaming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import CUDA
try:
    import radon_cuda
    CUDA_AVAILABLE = True
    logger.info("✅ CUDA acceleration loaded!")
except ImportError as e:
    CUDA_AVAILABLE = False
    logger.warning(f"⚠️ CUDA module not found: {e}")

from scipy.interpolate import griddata
if not CUDA_AVAILABLE:
    from skimage.transform import radon

# Processing functions
def radon_fun(f, alpha, R, ideal_image_size):
    if CUDA_AVAILABLE:
        f_flat = f.astype(np.float32)
        alpha_deg = alpha.astype(np.float32)
        p_radon = radon_cuda.radon_transform(f_flat, alpha_deg, R)
        p = p_radon * (R * 2 / ideal_image_size)
    else:
        p_radon = radon(f, theta=alpha, circle=True)
        p = p_radon * (R * 2 / ideal_image_size)
    
    s = np.arange(p.shape[0]) - p.shape[0] // 2
    s = s * (R * 2 / ideal_image_size)
    return p, s

def p_line_fun(alpha0, s0, PSI_1, PSI_2, R, L, tstart, tend, d, p_min):
    if CUDA_AVAILABLE:
        p_line = radon_cuda.calculate_p_line(
            float(alpha0), float(s0), PSI_1, PSI_2, L,
            float(R), float(tstart), float(tend), float(d), float(p_min)
        )
        return p_line
    else:
        ALPHA = (PSI_1 + PSI_2) / 2
        S = R * np.cos((PSI_2 - PSI_1) / 2)
        min_L = 0.01 * R
        valid_line_mask = L > min_L
        sin_term = np.sin(ALPHA - alpha0)
        sin_term_abs = np.abs(sin_term)
        denominator = (d * L - p_min) * sin_term_abs + p_min
        denominator = np.maximum(denominator, p_min / 100)
        p_line = d * p_min / denominator
        p_line = p_line * valid_line_mask
        p_line = np.minimum(p_line, d * 2)
        p_line = np.where(np.isfinite(p_line), p_line, 0)
        p_line = np.maximum(p_line, 0)
        return p_line

def AlphaS2Phi(ALPHA, S, PSI_1, PSI_2, p, R):
    if CUDA_AVAILABLE:
        alpha_min = float(np.min(ALPHA))
        alpha_max = float(np.max(ALPHA))
        s_min = float(np.min(S))
        s_max = float(np.max(S))
        p_interpolated = radon_cuda.grid_interpolation_gpu(
            p.astype(np.float32), PSI_1.astype(np.float32), PSI_2.astype(np.float32),
            float(R), alpha_min, alpha_max, s_min, s_max
        )
        return p_interpolated
    else:
        S_clipped = np.clip(S / R, -1.0, 1.0)
        PSI_1_convert = ALPHA - np.arccos(S_clipped)
        PSI_2_convert = ALPHA + np.arccos(S_clipped)
        points = np.column_stack([PSI_1_convert.ravel(), PSI_2_convert.ravel()])
        values = p.ravel()
        valid = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(values))
        p_q = griddata(points[valid], values[valid], (PSI_1, PSI_2), 
                       method='linear', fill_value=0.0)
        return p_q

def calculate_string_length(nail_sequence, num_nails, radius_cm):
    if len(nail_sequence) < 2:
        return 0.0
    radius_m = radius_cm / 100.0
    total_length = 0.0
    for i in range(len(nail_sequence) - 1):
        nail1 = nail_sequence[i]
        nail2 = nail_sequence[i + 1]
        angle1 = (nail1 / num_nails) * 2 * np.pi
        angle2 = (nail2 / num_nails) * 2 * np.pi
        angle_diff = abs(angle2 - angle1)
        chord_length = 2 * radius_m * np.sin(angle_diff / 2)
        total_length += chord_length
    return total_length

# GPU State
class GPUState:
    def __init__(self):
        self.is_busy = False
        self.current_job_id = None
        self.job_start_time = None
        self.total_jobs_completed = 0
        self.preprocessing_cache = {}
        
    def start_job(self, job_id: str):
        self.is_busy = True
        self.current_job_id = job_id
        self.job_start_time = time.time()
        logger.info(f"🚀 Started job: {job_id}")
        
    def finish_job(self):
        if self.is_busy:
            duration = time.time() - self.job_start_time
            logger.info(f"✅ Completed job {self.current_job_id} in {duration:.2f}s")
            self.total_jobs_completed += 1
        self.is_busy = False
        self.current_job_id = None
        self.job_start_time = None

gpu_state = GPUState()
start_time = time.time()

# Request Models
class PreprocessRequest(BaseModel):
    imageData: str
    num_nails: int
    image_resolution: int

class GenerateRequest(BaseModel):
    imageData: str
    params: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_busy: bool
    gpu_name: str
    current_job: Optional[str]
    jobs_completed: int
    uptime_seconds: float

# Health Endpoint
@app.get("/health")
async def health_check():
    return HealthResponse(
        status="healthy" if not gpu_state.is_busy else "busy",
        gpu_available=CUDA_AVAILABLE,
        gpu_busy=gpu_state.is_busy,
        gpu_name="RTX 3070" if CUDA_AVAILABLE else "CPU Only",
        current_job=gpu_state.current_job_id,
        jobs_completed=gpu_state.total_jobs_completed,
        uptime_seconds=time.time() - start_time
    )

# Preprocessing Endpoint
@app.post("/preprocess")
async def preprocess_image(request: PreprocessRequest):
    if gpu_state.is_busy:
        raise HTTPException(status_code=503, detail="GPU is busy")
    
    job_id = f"preprocess_{int(time.time() * 1000)}"
    cache_key = f"{request.num_nails}_{request.image_resolution}_{hash(request.imageData)}"
    
    if cache_key in gpu_state.preprocessing_cache:
        logger.info(f"✅ Using cached preprocessing for {cache_key}")
        cached_data = gpu_state.preprocessing_cache[cache_key]
        return {
            "status": "success",
            "message": "Preprocessing retrieved from cache",
            "cached": True,
            "cache_key": cache_key,
            **cached_data
        }
    
    gpu_state.start_job(job_id)
    
    try:
        logger.info(f"📥 Preprocessing: {request.num_nails} nails, {request.image_resolution}px")
        start = time.time()
        
        header, encoded = request.imageData.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img = img.resize((request.image_resolution, request.image_resolution), Image.Resampling.LANCZOS)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        BW_array = np.array(img).astype(np.float32) / 255.0
        
        R = 1.0
        ideal_image_size = request.image_resolution
        
        x = np.linspace(-R, R, ideal_image_size)
        y = np.linspace(-R, R, ideal_image_size)
        X, Y = np.meshgrid(x, y)
        circular_mask = X**2 + Y**2 <= R**2
        
        BW_array[~circular_mask] = 1.0
        f = 1 - BW_array
        f[~circular_mask] = 0
        
        Num_Nails = request.num_nails
        logger.info("  🔄 Computing Radon transform...")
        alpha_deg = np.linspace(0., 180., 3 * Num_Nails, endpoint=False)
        p, s = radon_fun(f, alpha_deg, R, ideal_image_size)
        
        ind_keep = np.abs(s) < R
        s, p = s[ind_keep], p[ind_keep, :]
        
        alpha_rad = np.deg2rad(alpha_deg)
        ALPHA, S = np.meshgrid(alpha_rad, s)
        L_alpha_s = 2 * np.sqrt(np.maximum(0, R**2 - S**2))
        p = p / (L_alpha_s + 1e-12)
        
        logger.info("  🔄 Interpolating to PSI space...")
        psi_1 = np.linspace(-np.pi, np.pi, Num_Nails + 1)
        psi_2 = np.linspace(0, 2 * np.pi, Num_Nails + 1)
        PSI_1, PSI_2 = np.meshgrid(psi_1, psi_2)
        
        angle_diff = np.abs(PSI_2 - PSI_1)
        L = 2 * R * np.sin(angle_diff / 2)
        L = np.maximum(L, 1e-12)
        
        p_interpolated = AlphaS2Phi(ALPHA, S, PSI_1, PSI_2, p, R)
        
        elapsed = time.time() - start
        logger.info(f"✅ Preprocessing complete in {elapsed:.2f}s")
        
        gpu_state.preprocessing_cache[cache_key] = {
            "p_interpolated": p_interpolated.copy(),
            "PSI_1": PSI_1,
            "PSI_2": PSI_2,
            "L": L,
            "psi_1": psi_1,
            "psi_2": psi_2
        }
        
        return {
            "status": "success",
            "message": "Preprocessing complete",
            "cached": False,
            "cache_key": cache_key,
            "processing_time": elapsed
        }
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gpu_state.finish_job()

# STREAMING GENERATOR (THE PROVEN V7 ALGORITHM WITH SSE)
async def generate_stream_logic(request: GenerateRequest):
    """Generator that yields SSE events - uses the proven v7 algorithm"""
    job_id = f"gen_{int(time.time() * 1000)}"
    
    try:
        gpu_state.start_job(job_id)
        
        params = request.params
        logger.info(f"📥 Generating with params: {params}")
        start_time_total = time.time()
        
        # Extract parameters
        Num_Nails = int(params.get('num_nails', 250))
        num_max_lines = int(params.get('num_strings', 4000))
        thread_thickness_mm = float(params.get('thread_thickness_mm', 0.25))
        circle_radius_cm = float(params.get('circle_radius_cm', 20))
        image_resolution = int(params.get('image_resolution', 300))
        
        # Decode image
        header, encoded = request.imageData.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img = img.resize((image_resolution, image_resolution), Image.Resampling.LANCZOS)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        BW_array = np.array(img).astype(np.float32) / 255.0
        
        R = 1.0
        
        x = np.linspace(-R, R, image_resolution)
        y = np.linspace(-R, R, image_resolution)
        X, Y = np.meshgrid(x, y)
        circular_mask = X**2 + Y**2 <= R**2
        
        BW_array[~circular_mask] = 1.0
        f = 1 - BW_array
        f[~circular_mask] = 0
        
        # Preprocessing
        cache_key = f"{Num_Nails}_{image_resolution}_{hash(request.imageData)}"
        use_cache = False
        
        if use_cache and cache_key in gpu_state.preprocessing_cache:
            logger.info("✅ Using cached preprocessing data")
            cached = gpu_state.preprocessing_cache[cache_key]
            p = cached["p_interpolated"].copy()
            PSI_1 = cached["PSI_1"]
            PSI_2 = cached["PSI_2"]
            L_cached = cached.get("L")
            psi_1 = cached.get("psi_1")
            psi_2 = cached.get("psi_2")
        else:
            logger.info("📊 Computing fresh preprocessing...")
            alpha_deg = np.linspace(0., 180., 3 * Num_Nails, endpoint=False)
            p, s = radon_fun(f, alpha_deg, R, f.shape[0])
            
            ind_keep = np.abs(s) < R
            s, p = s[ind_keep], p[ind_keep, :]
            
            alpha_rad = np.deg2rad(alpha_deg)
            ALPHA, S = np.meshgrid(alpha_rad, s)
            L_alpha_s = 2 * np.sqrt(np.maximum(0, R**2 - S**2))
            p = p / (L_alpha_s + 1e-12)
            
            psi_1 = np.linspace(-np.pi, np.pi, Num_Nails + 1)
            psi_2 = np.linspace(0, 2 * np.pi, Num_Nails + 1)
            PSI_1, PSI_2 = np.meshgrid(psi_1, psi_2)
            
            angle_diff = np.abs(PSI_2 - PSI_1)
            L_cached = 2 * R * np.sin(angle_diff / 2)
            L_cached = np.maximum(L_cached, 1e-12)
            
            p = AlphaS2Phi(ALPHA, S, PSI_1, PSI_2, p, R)
        
        # Greedy algorithm (PROVEN V7 VERSION)
        logger.info("🔄 Running greedy algorithm...")
        greedy_start = time.time()
        
        p = p.astype(np.float32)
        PSI_1 = PSI_1.astype(np.float32)
        PSI_2 = PSI_2.astype(np.float32)
        psi_1_np = psi_1.astype(np.float32)
        psi_2_np = psi_2.astype(np.float32)
        
        angle_diff = np.abs(PSI_2 - PSI_1)
        L = 2 * R * np.sin(angle_diff / 2)
        L = np.maximum(L, 1e-12).astype(np.float32)
        
        d = 0.036
        p_min = 0.00016
        tstart = 0.0014
        tend = 0.0161
        p_theshold = 0.0037
        
        psi_10 = np.zeros(num_max_lines)
        psi_20 = np.zeros(num_max_lines)
        nails_used = np.zeros(num_max_lines, dtype=int)
        size_p = p.shape
        row, col = 0, 0
        
        for i in range(num_max_lines):
            matlab_i = i + 1
            
            # THE PROVEN V7 ALTERNATING ALGORITHM
            if matlab_i % 2 == 1:
                if matlab_i == 1:
                    if CUDA_AVAILABLE:
                        p_max_val, ind = radon_cuda.find_max_and_index(p)
                        row, col = np.unravel_index(ind, size_p)
                    else:
                        p_max_val = float(np.nanmax(p))
                        ind = int(np.nanargmax(p))
                        row, col = np.unravel_index(ind, size_p)
                else:
                    if CUDA_AVAILABLE:
                        p_max_val, row = radon_cuda.find_max_in_column(p, col)
                    else:
                        p_max_val = float(np.nanmax(p[:, col]))
                        row = int(np.nanargmax(p[:, col]))
                
                psi_10[i] = float(psi_1_np[col])
                psi_20[i] = float(psi_2_np[row])
                current_nail = int(np.round(float(psi_2_np[row]) * Num_Nails / (2 * np.pi)))
            else:
                if CUDA_AVAILABLE:
                    p_max_val, col = radon_cuda.find_max_in_row(p, row)
                else:
                    p_max_val = float(np.nanmax(p[row, :]))
                    col = int(np.nanargmax(p[row, :]))
                
                psi_10[i] = float(psi_1_np[col])
                psi_20[i] = float(psi_2_np[row])
                psi_1_val = float(psi_1_np[col])
                current_nail = int(np.round(np.mod(psi_1_val + 2*np.pi, 2*np.pi) * Num_Nails / (2 * np.pi)))
            
            nails_used[i] = current_nail
            
            # 🎨 EMIT NEW LINE EVENT
            if i >= 1:
                event_data = {
                    "type": "new_line",
                    "start": int(nails_used[i-1]),
                    "end": int(nails_used[i])
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Force flush every 10 lines with SSE comment (prevents tunnel buffering)
                if i % 10 == 0:
                    yield ": ping\n\n"
            
            # Progress updates
            if matlab_i % 50 == 0:
                progress_data = {
                    "type": "progress",
                    "current": matlab_i,
                    "total": num_max_lines,
                    "percent": min(90, (matlab_i / num_max_lines) * 90)
                }
                yield f"data: {json.dumps(progress_data)}\n\n"
                logger.info(f"  Progress: {matlab_i}/{num_max_lines} lines")
            
            if p_max_val < p_theshold:
                logger.info(f'✅ Threshold reached at iteration {matlab_i}')
                break
            
            psi_1_current = psi_10[i]
            psi_2_current = psi_20[i]
            s0 = R * np.cos((psi_2_current - psi_1_current) / 2)
            alpha0 = (psi_1_current + psi_2_current) / 2
            
            p_line_theory = p_line_fun(alpha0, s0, PSI_1, PSI_2, R, L, tstart, tend, d, p_min)
            
            if CUDA_AVAILABLE:
                radon_cuda.subtract_p_line(p, p_line_theory)
            else:
                p = p - p_line_theory
                p = np.maximum(p, -0.1)
            
            if matlab_i >= 20 and matlab_i % 20 == 0:
                start_idx = max(0, i - 19)
                recent = nails_used[start_idx:i+1]
                if len(set(recent)) <= 3:
                    logger.warning(f"⚠️ Algorithm stuck in loop at iteration {matlab_i}")
                    break
        
        greedy_time = time.time() - greedy_start
        logger.info(f"  ⏱️ Greedy algorithm: {greedy_time:.2f}s")
        
        num_lines = i + 1
        logger.info(f'✅ Lines generated: {num_lines}')
        
        psi_10 = psi_10[:num_lines]
        psi_20 = psi_20[:num_lines]
        nails_used = nails_used[:num_lines]
        
        psi_1_converted = [np.mod(p + 2*np.pi, 2*np.pi) for p in psi_10]
        psi_2_converted = psi_20
        
        psi_1_number = [int(np.round(p * Num_Nails / (2 * np.pi))) for p in psi_1_converted]
        psi_2_number = [int(np.round(p * Num_Nails / (2 * np.pi))) for p in psi_2_converted]
        
        List = []
        for i in range(num_lines):
            matlab_i = i + 1
            if matlab_i % 2 == 0:
                List.append(psi_1_number[i])
            else:
                List.append(psi_2_number[i])
        
        List_1based = [n + 1 for n in List]
        total_length = calculate_string_length(List, Num_Nails, circle_radius_cm)
        
        total_time = time.time() - start_time_total
        
        # 🎨 EMIT FINAL SEQUENCE
        final_data = {
            "type": "final_sequence",
            "status": "success",
            "sequence": List_1based,
            "physical_info": {
                "radius_cm": circle_radius_cm,
                "radius_m": circle_radius_cm / 100.0,
                "thread_mm": thread_thickness_mm,
                "num_nails": Num_Nails,
                "num_lines": num_lines,
                "total_length_m": total_length
            },
            "processing_time": total_time,
            "gpu_used": "RTX 3070 (Home)" if CUDA_AVAILABLE else "CPU (Home)"
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        
        logger.info(f"✅ Completed job {job_id} in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        error_data = {
            "type": "error",
            "status": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    finally:
        gpu_state.finish_job()

# STREAMING ENDPOINT
@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    """Real-time SSE streaming endpoint"""
    if gpu_state.is_busy:
        raise HTTPException(status_code=503, detail="GPU is busy")
    
    return StreamingResponse(
        generate_stream_logic(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# LEGACY ENDPOINT
@app.post("/generate")
async def generate_string_art(request: GenerateRequest):
    """Legacy synchronous endpoint (uses same algorithm)"""
    if gpu_state.is_busy:
        raise HTTPException(status_code=503, detail="GPU is busy")
    
    result = None
    async for event in generate_stream_logic(request):
        if event.startswith("data: "):
            data = json.loads(event[6:])
            if data["type"] == "final_sequence":
                result = data
    
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Generation failed")

@app.get("/stats")
async def get_statistics():
    return {
        "total_jobs_completed": gpu_state.total_jobs_completed,
        "current_job": gpu_state.current_job_id,
        "is_busy": gpu_state.is_busy,
        "uptime_hours": (time.time() - start_time) / 3600,
        "gpu_name": "RTX 3070" if CUDA_AVAILABLE else "CPU Only",
        "cached_images": len(gpu_state.preprocessing_cache)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🎮 Home GPU Server - RTX 3070 with Real-Time Streaming")
    print("="*60)
    print(f"✅ CUDA: {'Available' if CUDA_AVAILABLE else 'Not available (CPU mode)'}")
    print("\n🚀 Endpoints:")
    print("  - /health (health check)")
    print("  - /generate_stream (SSE streaming - real-time animation)")
    print("  - /generate (legacy synchronous)")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")