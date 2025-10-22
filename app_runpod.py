# app_runpod.py - HTTP API version for RunPod deployment
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from scipy.interpolate import griddata
import base64
import io
import traceback
import time
import math

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

# Global cache for pre-processed data
preprocessing_cache = {}

# =============================================================================
# Helper Functions (same as your original)
# =============================================================================

def compute_radon_transform(image_array, num_angles):
    """Compute Radon transform using CUDA if available, else NumPy."""
    if CUDA_AVAILABLE:
        try:
            print(f"🚀 Using CUDA for Radon transform ({num_angles} angles)")
            return radon_cuda.radon_transform(image_array, num_angles)
        except Exception as e:
            print(f"⚠️ CUDA failed, falling back to NumPy: {e}")
    
    print(f"🐢 Using NumPy for Radon transform ({num_angles} angles)")
    theta = np.linspace(0., 180., num_angles, endpoint=False)
    return radon(image_array, theta=theta, circle=True)

def get_psi_grid(image_size, num_nails):
    """Pre-compute PSI grid for string art generation."""
    print(f"📊 Computing PSI grid: {image_size}x{image_size}, {num_nails} nails")
    start = time.time()
    
    angles = np.linspace(0, 2*np.pi, num_nails, endpoint=False)
    radius = image_size / 2.0
    
    nail_positions = np.column_stack([
        radius + (radius - 1) * np.cos(angles),
        radius + (radius - 1) * np.sin(angles)
    ])
    
    psi = np.zeros((num_nails, image_size), dtype=np.float32)
    
    for i in range(num_nails):
        for j in range(i + 1, num_nails):
            x1, y1 = nail_positions[i]
            x2, y2 = nail_positions[j]
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < 1e-6:
                continue
            
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad) % 180
            angle_idx = int(round(angle_deg / 180.0 * (image_size - 1)))
            angle_idx = np.clip(angle_idx, 0, image_size - 1)
            
            offset = ((x1 + x2) / 2.0 - radius) * np.cos(angle_rad) + \
                     ((y1 + y2) / 2.0 - radius) * np.sin(angle_rad)
            
            center = image_size / 2.0
            offset_idx = int(round(center + offset))
            offset_idx = np.clip(offset_idx, 0, image_size - 1)
            
            psi[i, angle_idx] += 1.0 / (1.0 + abs(offset_idx - center) / radius)
            psi[j, angle_idx] += 1.0 / (1.0 + abs(offset_idx - center) / radius)
    
    elapsed = time.time() - start
    print(f"✅ PSI grid computed in {elapsed:.2f}s")
    
    return psi

# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'cuda_available': CUDA_AVAILABLE
    }), 200

@app.route('/preprocess', methods=['POST'])
def preprocess_image():
    """Pre-process image (compute Radon transform and PSI grid)."""
    try:
        data = request.json
        image_data = data['imageData']
        num_nails = data['num_nails']
        image_resolution = data['image_resolution']
        
        print(f"🔄 Pre-processing: {num_nails} nails, {image_resolution}x{image_resolution}")
        
        # Decode image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((image_resolution, image_resolution), Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Compute Radon transform
        radon_image = compute_radon_transform(image_array, image_resolution)
        
        # Compute PSI grid
        psi = get_psi_grid(image_resolution, num_nails)
        
        # Cache results
        cache_key = f"{num_nails}_{image_resolution}"
        preprocessing_cache[cache_key] = {
            'radon': radon_image,
            'psi': psi,
            'image_array': image_array
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Pre-processing complete',
            'cache_key': cache_key
        }), 200
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_pattern():
    """Generate string art pattern."""
    try:
        data = request.json
        params = data['params']
        cache_key = data.get('cache_key')
        
        num_nails = params['num_nails']
        num_strings = params['num_strings']
        image_resolution = params['image_resolution']
        
        print(f"🎨 Generating: {num_strings} strings, {num_nails} nails")
        
        # Get cached data or re-process
        if cache_key and cache_key in preprocessing_cache:
            print("✅ Using cached pre-processed data")
            cached = preprocessing_cache[cache_key]
            radon_image = cached['radon']
            psi = cached['psi']
        else:
            print("⚠️ No cache, re-processing")
            image_data = data['imageData']
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            image = image.resize((image_resolution, image_resolution), Image.Resampling.LANCZOS)
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            radon_image = compute_radon_transform(image_array, image_resolution)
            psi = get_psi_grid(image_resolution, num_nails)
        
        # Generate string sequence
        sequence = []
        current_nail = 0
        radon_copy = radon_image.copy()
        
        for string_idx in range(num_strings):
            # Find best next nail
            scores = np.dot(psi, radon_copy)
            scores[current_nail] = -np.inf  # Can't connect to self
            
            next_nail = np.argmax(scores)
            sequence.append(int(current_nail))
            
            # Update Radon image (subtract contribution)
            angle_diff = abs(next_nail - current_nail)
            if angle_diff > num_nails / 2:
                angle_diff = num_nails - angle_diff
            
            angle_rad = (angle_diff / num_nails) * np.pi
            angle_deg = np.degrees(angle_rad) % 180
            angle_idx = int(round(angle_deg / 180.0 * (image_resolution - 1)))
            angle_idx = np.clip(angle_idx, 0, image_resolution - 1)
            
            radon_copy[:, angle_idx] *= 0.9  # Reduce contribution
            
            current_nail = next_nail
            
            # Progress updates every 10%
            if (string_idx + 1) % (num_strings // 10) == 0:
                progress = ((string_idx + 1) / num_strings) * 100
                print(f"⏳ Progress: {progress:.0f}%")
        
        sequence.append(int(current_nail))  # Close the loop
        
        # Calculate physical info
        radius_cm = params['circle_radius_cm']
        thread_mm = params['thread_thickness_mm']
        
        total_length_m = 0
        angles = np.linspace(0, 2*np.pi, num_nails, endpoint=False)
        radius_m = radius_cm / 100.0
        
        for i in range(len(sequence) - 1):
            nail1 = sequence[i]
            nail2 = sequence[i + 1]
            
            x1 = radius_m * np.cos(angles[nail1])
            y1 = radius_m * np.sin(angles[nail1])
            x2 = radius_m * np.cos(angles[nail2])
            y2 = radius_m * np.sin(angles[nail2])
            
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length_m += length
        
        physical_info = {
            'radius_cm': radius_cm,
            'radius_m': radius_m,
            'thread_mm': thread_mm,
            'num_nails': num_nails,
            'num_lines': len(sequence) - 1,
            'total_length_m': round(total_length_m, 2)
        }
        
        print(f"✅ Generation complete! {len(sequence)} pins, {total_length_m:.2f}m string")
        
        return jsonify({
            'status': 'success',
            'sequence': sequence,
            'physical_info': physical_info
        }), 200
        
    except Exception as e:
        print(f"❌ Error in generation: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = 8000
    print(f"\n{'='*60}")
    print(f"🎨 String Art GPU Backend")
    print(f"{'='*60}")
    print(f"🌐 Server: http://0.0.0.0:{port}")
    print(f"🔥 CUDA: {'Enabled ✅' if CUDA_AVAILABLE else 'Disabled ❌'}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)