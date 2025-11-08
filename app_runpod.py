# app_runpod.py - HTTP API version for RunPod deployment (CORRECTED CACHE LOGIC)
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from scipy.interpolate import griddata
import base64
import io
import traceback
import time
import math
import hashlib

# Try to import CUDA module
try:
    import radon_cuda
    CUDA_AVAILABLE = True
    print("✅ CUDA acceleration loaded successfully!")
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"⚠️ CUDA module not found: {e}")
    print("Falling back to NumPy (CPU only)")

# Fallback imports if CUDA not available
if not CUDA_AVAILABLE:
    from skimage.transform import radon

app = Flask(__name__)

preprocessing_cache = {}

# =============================================================================
# Helper Functions (Copied from working app_cuda.py)
# =============================================================================
# radon_fun, p_line_fun, calculate_string_length, AlphaS2Phi remain unchanged...
def radon_fun(f, alpha, R, ideal_image_size):
    """Calculate radon transform"""
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
    """Calculates radon transform of line - CUDA accelerated"""
    if CUDA_AVAILABLE:
        p_line = radon_cuda.calculate_p_line( float(alpha0), float(s0), PSI_1, PSI_2, L, float(R), float(tstart), float(tend), float(d), float(p_min) )
        return p_line
    else:
        ALPHA = (PSI_1 + PSI_2) / 2
        S = R * np.cos((PSI_2 - PSI_1) / 2)
        min_L = 0.01 * R; valid_line_mask = L > min_L
        sin_term = np.sin(ALPHA - alpha0); sin_term_abs = np.abs(sin_term)
        denominator = (d * L - p_min) * sin_term_abs + p_min
        denominator = np.maximum(denominator, p_min / 100)
        p_line = d * p_min / denominator
        p_line = p_line * valid_line_mask; p_line = np.minimum(p_line, d * 2)
        p_line = np.where(np.isfinite(p_line), p_line, 0); p_line = np.maximum(p_line, 0)
        return p_line

def calculate_string_length(nail_sequence, num_nails, radius_cm):
    """Calculate total length of string needed in meters"""
    if len(nail_sequence) < 2: return 0.0
    radius_m = radius_cm / 100.0; total_length = 0.0
    for i in range(len(nail_sequence) - 1):
        nail1 = nail_sequence[i] - 1; nail2 = nail_sequence[i + 1] - 1
        angle1 = (nail1 / num_nails) * 2 * np.pi; angle2 = (nail2 / num_nails) * 2 * np.pi
        angle_diff = abs(angle2 - angle1)
        if angle_diff > np.pi: angle_diff = 2 * np.pi - angle_diff # Use shorter angle
        chord_length = 2 * radius_m * np.sin(angle_diff / 2)
        total_length += chord_length
    return total_length

def AlphaS2Phi(ALPHA, S, PSI_1, PSI_2, p, R):
    """Interpolation - uses CUDA when available"""
    use_cuda = CUDA_AVAILABLE
    if use_cuda:
        alpha_min=float(np.min(ALPHA)); alpha_max=float(np.max(ALPHA))
        s_min=float(np.min(S)); s_max=float(np.max(S))
        p_interpolated = radon_cuda.grid_interpolation_gpu( p.astype(np.float32), PSI_1.astype(np.float32), PSI_2.astype(np.float32), float(R), alpha_min, alpha_max, s_min, s_max )
        return p_interpolated
    else:
        S_clipped = np.clip(S / R, -1.0, 1.0)
        PSI_1_convert = ALPHA - np.arccos(S_clipped); PSI_2_convert = ALPHA + np.arccos(S_clipped)
        points = np.column_stack([PSI_1_convert.ravel(), PSI_2_convert.ravel()])
        values = p.ravel()
        valid = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(values))
        p_q = griddata(points[valid], values[valid], (PSI_1, PSI_2), method='linear', fill_value=0.0)
        return p_q

# =============================================================================
# Background Pre-Processing Function (for API)
# =============================================================================
def preprocess_image(image_bytes, num_nails, image_resolution, invert_colors=False):
    """Pre-compute Radon transform and interpolation"""
    global preprocessing_cache

    try:
        print(f"\n📄 Starting pre-processing...")
        t_total_start = time.time()

        # --- Calculate image hash ---
        img_hash = hashlib.sha256(image_bytes).hexdigest()[:16] # Use first 16 chars of hash
        cache_key = f"{img_hash}_{num_nails}_{image_resolution}_{invert_colors}" # <-- Include invert_colors in cache key
        
        # --- Check cache FIRST ---
        if cache_key in preprocessing_cache:
            print(f"✅ Pre-processing found in cache for key: {cache_key}")
            return {'status': 'success', 'message': 'Pre-processing complete (cached)', 'cache_key': cache_key}

        print(f"🔧 Pre-processing for new key: {cache_key}")
        
        d, p_min, tstart, tend = 0.036, 0.00016, 0.0014, 0.0161
        R = 1.0
        ideal_image_size = int(image_resolution)
        Num_Nails = int(num_nails)
        
        print("📸 Loading and processing image...")
        t_start = time.time()
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        BW = img.resize((ideal_image_size, ideal_image_size)).transpose(Image.FLIP_TOP_BOTTOM)
        BW_array = np.array(BW) / 255.0
        t_image = time.time() - t_start
        print(f"  ⏱️ Image loading: {t_image:.3f}s")
        
        print("🎭 Creating circular mask...")
        t_start = time.time()
        x = np.linspace(-R, R, ideal_image_size); y = np.linspace(-R, R, ideal_image_size)
        X, Y = np.meshgrid(x, y); circular_mask = X**2 + Y**2 <= R**2
        BW_array[~circular_mask] = 1.0

        # Handle inverted colors mode
        if invert_colors:
            f = BW_array  # Keep bright values high (algorithm will match bright areas)
            print("🎨 INVERTED MODE: Targeting bright areas (white-on-black)")
        else:
            f = 1 - BW_array  # Keep dark values high (algorithm will match dark areas)
            print("🎨 NORMAL MODE: Targeting dark areas (black-on-white)")

        f[~circular_mask] = 0
        t_mask = time.time() - t_start
        print(f"  ⏱️ Mask creation: {t_mask:.3f}s")
        
        print("🔄 Computing Radon transform...")
        t_start = time.time()
        alpha_deg = np.linspace(0., 180., 3 * Num_Nails, endpoint=False)
        p, s = radon_fun(f, alpha_deg, R, ideal_image_size)
        t_radon = time.time() - t_start
        print(f"  ⏱️ Radon transform: {t_radon:.3f}s")
        
        print("🔎 Filtering radon data...")
        t_start = time.time()
        ind_keep = np.abs(s) < R; s, p = s[ind_keep], p[ind_keep, :]
        alpha_rad = np.deg2rad(alpha_deg); ALPHA, S = np.meshgrid(alpha_rad, s)
        L_alpha_s = 2 * np.sqrt(np.maximum(0, R**2 - S**2)); p = p / (L_alpha_s + 1e-12)
        t_filter = time.time() - t_start
        print(f"  ⏱️ Filtering: {t_filter:.3f}s")
        
        print("🔧 Creating PSI grid...")
        t_start = time.time()
        psi_1 = np.linspace(-np.pi, np.pi, Num_Nails + 1)
        psi_2 = np.linspace(0, 2 * np.pi, Num_Nails + 1)
        PSI_1, PSI_2 = np.meshgrid(psi_1, psi_2)
        angle_diff = np.abs(PSI_2 - PSI_1)
        L = 2 * R * np.sin(angle_diff / 2); L = np.maximum(L, 1e-12)
        t_psi = time.time() - t_start
        print(f"  ⏱️ PSI grid: {t_psi:.3f}s")
        
        print("🔀 Interpolating to PSI coordinates...")
        t_start = time.time()
        p_interpolated = AlphaS2Phi(ALPHA, S, PSI_1, PSI_2, p, R)
        p_interpolated = p_interpolated.astype(np.float32)
        t_interp = time.time() - t_start
        print(f"  ⏱️ Interpolation: {t_interp:.3f}s")
        
        print("💾 Caching results...")
        # Cache key already calculated above
        preprocessing_cache[cache_key] = {
            'p': p_interpolated, 'PSI_1': PSI_1.astype(np.float32),
            'PSI_2': PSI_2.astype(np.float32), 'L': L.astype(np.float32),
            'psi_1': psi_1.astype(np.float32), 'psi_2': psi_2.astype(np.float32),
            'R': R, 'cache_key': cache_key, 'Num_Nails': Num_Nails
        }
        
        t_total = time.time() - t_total_start
        print(f"\n✅ Pre-processing complete! Total time: {t_total:.3f}s")
        
        return {'status': 'success', 'message': 'Pre-processing complete', 'cache_key': cache_key}
        
    except Exception as e:
        print(f"❌ Pre-processing error: {e}")
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}

# =============================================================================
# Main Generation Function (for API)
# =============================================================================
def generate_pattern(data):
    """Main algorithm with CUDA acceleration (API version)"""
    global preprocessing_cache
    
    try:
        params = data['params']
        image_data_url = data['imageData']
        
        Num_Nails = int(params.get('num_nails', 250))
        num_max_lines = int(params.get('num_strings', 4000))
        circle_radius_cm = float(params.get('circle_radius_cm', 30))
        thread_thickness_mm = float(params.get('thread_thickness_mm', 0.5))
        ideal_image_size = int(params.get('image_resolution', 300))
        invert_colors = bool(params.get('invert_colors', False))

        d, p_min, tstart, tend = 0.036, 0.00016, 0.0014, 0.0161
        p_theshold = 0.0037

        print(f"\n{'='*60}")
        print(f"String Art Generation")
        print(f"{'='*60}")
        print(f"  Nails: {Num_Nails}, Max Lines: {num_max_lines}")
        print(f"  Resolution: {ideal_image_size}x{ideal_image_size}")
        print(f"  Acceleration: {'CUDA (Native C++)' if CUDA_AVAILABLE else 'CPU (NumPy/SciPy)'}")

        # --- Calculate image hash and cache key ---
        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        cache_key = f"{img_hash}_{Num_Nails}_{ideal_image_size}_{invert_colors}" # <-- Include invert_colors in cache key
        
        use_cache = False
        
        if cache_key in preprocessing_cache:
            cached = preprocessing_cache[cache_key]
            use_cache = True
            print(f"🚀 Using pre-processed data from cache! Key: {cache_key}")
            p = cached['p'].copy() # <-- Use a copy to avoid modifying cache
            PSI_1 = cached['PSI_1']; PSI_2 = cached['PSI_2']; L = cached['L']
            psi_1 = cached['psi_1']; psi_2 = cached['psi_2']; R = cached['R']
        
        if not use_cache:
            print(f"⚙️ No cache found for key {cache_key}. Running pre-processing first...")

            # Run the preprocessing step
            preprocess_result = preprocess_image(image_bytes, Num_Nails, ideal_image_size, invert_colors)
            if preprocess_result['status'] == 'error':
                return preprocess_result # Propagate error

            # Now load from cache (which must exist now)
            cached = preprocessing_cache[cache_key]
            p = cached['p'].copy() # <-- Use a copy
            PSI_1 = cached['PSI_1']; PSI_2 = cached['PSI_2']; L = cached['L']
            psi_1 = cached['psi_1']; psi_2 = cached['psi_2']; R = cached['R']
        
        print("🎯 Starting optimized greedy algorithm...")
        
        t_greedy_start = time.time()
        psi_10, psi_20 = [], []
        nails_used = []
        size_p = p.shape
        row, col = 0, 0
        
        p = p.astype(np.float32) # Ensure correct type
        
        for i in range(num_max_lines):
            matlab_i = i + 1
            
            if matlab_i % 2 == 1: # Odd iteration
                if matlab_i == 1:
                    # Find global max on first iteration
                    if CUDA_AVAILABLE: p_max_val, ind = radon_cuda.find_max_and_index(p); row, col = np.unravel_index(ind, size_p)
                    else: p_max_val = float(np.nanmax(p)); ind = int(np.nanargmax(p)); row, col = np.unravel_index(ind, size_p)
                else:
                    # Find max in previous column
                    if CUDA_AVAILABLE: p_max_val, row = radon_cuda.find_max_in_column(p, col)
                    else: p_max_val = float(np.nanmax(p[:, col])); row = int(np.nanargmax(p[:, col]))
                
                psi_10.append(float(psi_1[col])); psi_20.append(float(psi_2[row]))
                current_nail = int(np.round(float(psi_2[row]) * Num_Nails / (2 * np.pi))) # Nail index from psi_2
            else: # Even iteration
                # Find max in previous row
                if CUDA_AVAILABLE: p_max_val, col = radon_cuda.find_max_in_row(p, row)
                else: p_max_val = float(np.nanmax(p[row, :])); col = int(np.nanargmax(p[row, :]))
                
                psi_10.append(float(psi_1[col])); psi_20.append(float(psi_2[row]))
                psi_1_val = float(psi_1[col])
                current_nail = int(np.round(np.mod(psi_1_val + 2*np.pi, 2*np.pi) * Num_Nails / (2 * np.pi))) # Nail index from psi_1 (wrapped)

            nails_used.append(current_nail) # 0-based nail index
            
            if p_max_val < p_theshold:
                print(f'✅ Threshold reached at iteration {matlab_i}')
                break
            
            psi_1_current = psi_10[-1]; psi_2_current = psi_20[-1]
            s0 = R * np.cos((psi_2_current - psi_1_current) / 2)
            alpha0 = (psi_1_current + psi_2_current) / 2
            
            p_line_theory = p_line_fun(alpha0, s0, PSI_1, PSI_2, R, L, tstart, tend, d, p_min)
            
            if CUDA_AVAILABLE: radon_cuda.subtract_p_line(p, p_line_theory)
            else: p = p - p_line_theory; p = np.maximum(p, -0.1)
            
            # Check for loops
            if matlab_i >= 20 and matlab_i % 20 == 0:
                recent = nails_used[-20:]
                if len(set(recent)) <= 3: print(f"⚠️ WARNING: Algorithm stuck in loop at iteration {matlab_i}"); break
            
            if matlab_i % 1000 == 0: print(f"  Progress: {matlab_i}/{num_max_lines} lines, p_max={p_max_val:.6f}")
        
        t_greedy = time.time() - t_greedy_start
        print(f"  ⏱️ Greedy algorithm: {t_greedy:.2f}s")
        
        print("📊 Generating final output...")
        num_lines = len(psi_10)
        
        if num_lines > 0:
            psi_1_converted = [np.mod(p + 2*np.pi, 2*np.pi) for p in psi_10]
            psi_2_converted = psi_20
            
            psi_1_number = [int(np.round(p * Num_Nails / (2 * np.pi))) for p in psi_1_converted]
            psi_2_number = [int(np.round(p * Num_Nails / (2 * np.pi))) for p in psi_2_converted]
            
            List = [] # This will be 0-based nail indices
            for i in range(num_lines):
                matlab_i = i + 1
                if matlab_i % 2 == 0: List.append(psi_1_number[i])
                else: List.append(psi_2_number[i])
            
            List_1based = [n + 1 for n in List] # Convert to 1-based for display/output
            total_length = calculate_string_length(List_1based, Num_Nails, circle_radius_cm)
            
            result = {
                'status': 'success',
                'sequence': List_1based,
                'physical_info': {
                    'radius_cm': circle_radius_cm, 'radius_m': circle_radius_cm / 100.0,
                    'thread_mm': thread_thickness_mm, 'num_nails': Num_Nails,
                    'num_lines': num_lines, 'total_length_m': total_length
                }
            }
            print("✨ Done!\n")
            return result
        else:
            return {'status': 'error', 'message': 'Generation failed to produce lines'}
        
    except Exception as e:
        print(f"❌ ERROR in generate_pattern: {e}")
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}

# =============================================================================
# API Endpoints (Not used by handler, but good for testing)
# =============================================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'cuda_available': CUDA_AVAILABLE}), 200

@app.route('/generate_pattern', methods=['POST'])
def generate_endpoint():
    result = generate_pattern(request.json)
    return jsonify(result), 500 if result['status'] == 'error' else 200

@app.route('/preprocess_image', methods=['POST'])
def preprocess_endpoint():
    data = request.json
    try:
        header, encoded = data['imageData'].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        num_nails = int(data['num_nails']); image_resolution = int(data['image_resolution'])
        result = preprocess_image(image_bytes, num_nails, image_resolution)
        return jsonify(result), 500 if result['status'] == 'error' else 200
    except Exception as e:
        print(f"❌ Error in preprocess_endpoint: {e}"); traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = 8000
    print(f"\n{'='*60}\n🎨 String Art GPU Backend (HTTP Test Server)\n{'='*60}")
    print(f"🌐 Server: http://0.0.0.0:{port}")
    print(f"🔥 CUDA: {'Enabled ✅' if CUDA_AVAILABLE else 'Disabled ❌'}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=False)