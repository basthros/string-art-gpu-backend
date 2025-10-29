# FILENAME: app_cuda.py

# STEP 1: Apply the gevent patch. This MUST be the first line.
from gevent import monkey
monkey.patch_all()

# STEP 2: Import everything else.
from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO
import numpy as np
from PIL import Image
from scipy.interpolate import griddata
import base64
import io
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import traceback
import time
import math

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib import colors

# Try to import CUDA module
try:
    import radon_cuda
    CUDA_AVAILABLE = True
    print("✅ CUDA acceleration loaded successfully!")
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"⚠️ CUDA module not found: {e}")
    print("Falling back to NumPy (CPU only)")
    print("Build CUDA module with: python setup.py build_ext --inplace")

# Fallback imports if CUDA not available
if not CUDA_AVAILABLE:
    from skimage.transform import radon

# STEP 3: Create the application instance.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='gevent', max_http_buffer_size=50 * 1024 * 1024, 
                    ping_timeout=120, ping_interval=60)

# Global cancel flags
cancel_flags = {}

# Global cache for pre-processed data (per session)
preprocessing_cache = {}
preprocessing_in_progress = {}

# =============================================================================
# Helper Functions
# =============================================================================
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
        p_line = radon_cuda.calculate_p_line(
            float(alpha0), float(s0),
            PSI_1, PSI_2, L,
            float(R), float(tstart), float(tend),
            float(d), float(p_min)
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

def calculate_string_length(nail_sequence, num_nails, radius_cm):
    """Calculate total length of string needed in meters"""
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

def AlphaS2Phi(ALPHA, S, PSI_1, PSI_2, p, R):
    """Interpolation - uses CUDA when available"""
    # TEMPORARY: Force CPU to test
    use_cuda = CUDA_AVAILABLE  # Changed from: False
    
    if use_cuda:
        # Use CUDA acceleration
        alpha_min = float(np.min(ALPHA))
        alpha_max = float(np.max(ALPHA))
        s_min = float(np.min(S))
        s_max = float(np.max(S))
        
        p_interpolated = radon_cuda.grid_interpolation_gpu(
            p.astype(np.float32),
            PSI_1.astype(np.float32),
            PSI_2.astype(np.float32),
            float(R),
            alpha_min, alpha_max,
            s_min, s_max
        )
        return p_interpolated
    else:
        # CPU fallback
        S_clipped = np.clip(S / R, -1.0, 1.0)
        PSI_1_convert = ALPHA - np.arccos(S_clipped)
        PSI_2_convert = ALPHA + np.arccos(S_clipped)
        
        points = np.column_stack([PSI_1_convert.ravel(), PSI_2_convert.ravel()])
        values = p.ravel()
        
        valid = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]) | np.isnan(values))
        
        p_q = griddata(points[valid], values[valid], (PSI_1, PSI_2), 
                       method='linear', fill_value=0.0)
        
        return p_q

def generate_printable_template(num_nails, radius_cm):
    """Generate multi-page printable template with assembly guide"""
    
    print(f"📄 Generating PDF template: {num_nails} nails, {radius_cm}cm radius")
    
    PAGE_WIDTH = 8.5 * inch
    PAGE_HEIGHT = 11 * inch
    PAGE_WIDTH_INCHES = 8.5
    PAGE_HEIGHT_INCHES = 11
    
    radius_inches = radius_cm / 2.54
    diameter_inches = radius_inches * 2
    
    # Calculate grid size
    pages_x = int(math.ceil(diameter_inches / PAGE_WIDTH_INCHES)) 
    pages_y = int(math.ceil(diameter_inches / PAGE_HEIGHT_INCHES))
    
    total_width_inches = pages_x * PAGE_WIDTH_INCHES
    total_height_inches = pages_y * PAGE_HEIGHT_INCHES
    center_x_total = total_width_inches / 2
    center_y_total = total_height_inches / 2
    
    # =========================================================================
    # STEP 1: Determine which pages have content
    # =========================================================================
    pages_with_content = set()
    
    for nail_idx in range(num_nails):
        angle = (nail_idx / num_nails) * 2 * math.pi
        nail_x_inches = center_x_total + radius_inches * math.cos(angle)
        nail_y_inches = center_y_total + radius_inches * math.sin(angle)
        
        col = int(nail_x_inches / PAGE_WIDTH_INCHES)
        row = int(nail_y_inches / PAGE_HEIGHT_INCHES)
        
        if 0 <= col < pages_x and 0 <= row < pages_y:
            pages_with_content.add((row, col))
    
    print(f"  Total grid: {pages_x} x {pages_y} = {pages_x * pages_y} pages")
    print(f"  Pages with content: {len(pages_with_content)} pages")
    
    buffer = io.BytesIO()
    c = pdf_canvas.Canvas(buffer, pagesize=letter)
    
    # =========================================================================
    # Page 1: Assembly Guide
    # =========================================================================
    MARGIN = 0.5 * inch
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN, PAGE_HEIGHT - MARGIN, "String Art Template - Assembly Guide")
    
    c.setFont("Helvetica", 12)
    y_pos = PAGE_HEIGHT - MARGIN - 30
    c.drawString(MARGIN, y_pos, f"Circle Diameter: {radius_cm * 2:.1f} cm ({diameter_inches:.1f} inches)")
    y_pos -= 20
    c.drawString(MARGIN, y_pos, f"Number of Nails: {num_nails}")
    y_pos -= 20
    c.drawString(MARGIN, y_pos, f"Template Pages: {len(pages_with_content)} pages (from {pages_x} × {pages_y} grid)")
    
    y_pos -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y_pos, "Assembly Instructions:")
    
    y_pos -= 25
    c.setFont("Helvetica", 11)
    instructions = [
        "1. Print all pages at 100% scale (DO NOT scale to fit)",
        "2. Use blank paper sheets for empty grid positions shown below",
        "3. Arrange pages according to the grid (with blank spacers)",
        "4. Tape pages together on the back side",
        "5. Poke holes at the red crosshair marks",
        "6. Mount on backing board and insert nails"
    ]
    
    for instruction in instructions:
        c.drawString(MARGIN, y_pos, instruction)
        y_pos -= 18
    
    # Assembly grid
    y_pos -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN, y_pos, "Page Assembly Grid:")
    y_pos -= 15
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, y_pos, "(Gray cells = blank spacer paper needed)")
    
    y_pos -= 25
    
    # Calculate preview cell size maintaining aspect ratio
    max_grid_width = PAGE_WIDTH - 2 * MARGIN - 40
    max_grid_height = y_pos - MARGIN - 40
    
    preview_cell_width = min(70, max_grid_width / pages_x)
    preview_cell_height = preview_cell_width * (PAGE_HEIGHT_INCHES / PAGE_WIDTH_INCHES)  # Maintain paper aspect ratio
    
    # Adjust if height is too large
    if preview_cell_height * pages_y > max_grid_height:
        preview_cell_height = max_grid_height / pages_y
        preview_cell_width = preview_cell_height * (PAGE_WIDTH_INCHES / PAGE_HEIGHT_INCHES)
    
    grid_start_x = MARGIN + 20
    grid_start_y = y_pos - (pages_y * preview_cell_height) - 20
    
    # Create page number map
    page_number_map = {}
    page_num = 2
    for display_row in range(pages_y):
        for col in range(pages_x):
            coord_row = pages_y - 1 - display_row
            if (coord_row, col) in pages_with_content:
                page_number_map[(coord_row, col)] = page_num
                page_num += 1
    
    # Draw grid with CORRECT ASPECT RATIO
    for display_row in range(pages_y):
        for col in range(pages_x):
            cell_x = grid_start_x + col * preview_cell_width
            cell_y = grid_start_y + (pages_y - 1 - display_row) * preview_cell_height
            coord_row = pages_y - 1 - display_row
            
            has_content = (coord_row, col) in pages_with_content
            
            # Fill background
            if not has_content:
                c.setFillColor(colors.lightgrey)
                c.rect(cell_x, cell_y, preview_cell_width, preview_cell_height, stroke=0, fill=1)
            
            # Draw border
            c.setStrokeColor(colors.black)
            c.setFillColor(colors.black)
            c.setLineWidth(1)
            c.rect(cell_x, cell_y, preview_cell_width, preview_cell_height, stroke=1, fill=0)
            
            if has_content:
                page_left_inches = col * PAGE_WIDTH_INCHES
                page_bottom_inches = coord_row * PAGE_HEIGHT_INCHES
                page_right_inches = (col + 1) * PAGE_WIDTH_INCHES
                page_top_inches = (coord_row + 1) * PAGE_HEIGHT_INCHES
                
                # Draw circle preview with CORRECT SCALING
                c.setStrokeColor(colors.blue)
                c.setLineWidth(1.5)
                
                # CRITICAL: Use different scale factors for X and Y
                scale_x = preview_cell_width / PAGE_WIDTH_INCHES
                scale_y = preview_cell_height / PAGE_HEIGHT_INCHES
                
                num_circle_segments = 360
                for i in range(num_circle_segments):
                    angle1 = (i / num_circle_segments) * 2 * math.pi
                    angle2 = ((i + 1) / num_circle_segments) * 2 * math.pi
                    
                    x1_inches = center_x_total + radius_inches * math.cos(angle1)
                    y1_inches = center_y_total + radius_inches * math.sin(angle1)
                    x2_inches = center_x_total + radius_inches * math.cos(angle2)
                    y2_inches = center_y_total + radius_inches * math.sin(angle2)
                    
                    margin = 0.5
                    in_x = ((page_left_inches - margin <= x1_inches <= page_right_inches + margin) or
                           (page_left_inches - margin <= x2_inches <= page_right_inches + margin))
                    in_y = ((page_bottom_inches - margin <= y1_inches <= page_top_inches + margin) or
                           (page_bottom_inches - margin <= y2_inches <= page_top_inches + margin))
                    
                    if in_x and in_y:
                        x1_in_page = x1_inches - page_left_inches
                        y1_in_page = y1_inches - page_bottom_inches
                        x2_in_page = x2_inches - page_left_inches
                        y2_in_page = y2_inches - page_bottom_inches
                        
                        x1_preview = cell_x + x1_in_page * scale_x
                        y1_preview = cell_y + y1_in_page * scale_y
                        x2_preview = cell_x + x2_in_page * scale_x
                        y2_preview = cell_y + y2_in_page * scale_y
                        
                        c.line(x1_preview, y1_preview, x2_preview, y2_preview)
                
                # Draw page number
                pnum = page_number_map[(coord_row, col)]
                c.setFont("Helvetica-Bold", 8)
                c.setFillColor(colors.black)
                c.drawCentredString(cell_x + preview_cell_width/2, cell_y + 5, f"P{pnum}")
            else:
                # Blank cell
                c.setFont("Helvetica", 7)
                c.setFillColor(colors.grey)
                c.drawCentredString(cell_x + preview_cell_width/2, cell_y + preview_cell_height/2 - 3, "BLANK")
    
    c.showPage()
    
    # =========================================================================
    # Pages 2+: Only generate pages with content
    # =========================================================================
    for (coord_row, col), page_num in sorted(page_number_map.items(), key=lambda x: x[1]):
        page_left_inches = col * PAGE_WIDTH_INCHES
        page_bottom_inches = coord_row * PAGE_HEIGHT_INCHES
        page_right_inches = (col + 1) * PAGE_WIDTH_INCHES
        page_top_inches = (coord_row + 1) * PAGE_HEIGHT_INCHES
        
        display_row = pages_y - 1 - coord_row
        
        # Page info
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        c.drawString(10, PAGE_HEIGHT - 15, f"Page {page_num}/{len(pages_with_content) + 1}")
        c.drawString(10, PAGE_HEIGHT - 25, f"Grid: Row {display_row + 1}, Col {col + 1}")
        c.setFillColor(colors.black)
        
        # Draw nails
        for nail_idx in range(num_nails):
            angle = (nail_idx / num_nails) * 2 * math.pi
            
            nail_x_inches = center_x_total + radius_inches * math.cos(angle)
            nail_y_inches = center_y_total + radius_inches * math.sin(angle)
            
            if (page_left_inches <= nail_x_inches <= page_right_inches and 
                page_bottom_inches <= nail_y_inches <= page_top_inches):
                
                nail_x_page = (nail_x_inches - page_left_inches) * inch
                nail_y_page = (nail_y_inches - page_bottom_inches) * inch
                
                # Crosshair
                marker_size = 10
                c.setStrokeColor(colors.red)
                c.setLineWidth(1.5)
                c.line(nail_x_page - marker_size, nail_y_page, nail_x_page + marker_size, nail_y_page)
                c.line(nail_x_page, nail_y_page - marker_size, nail_x_page, nail_y_page + marker_size)
                c.circle(nail_x_page, nail_y_page, 4, stroke=1, fill=0)
                
                # Label (12 pixels away)
                dx = nail_x_inches - center_x_total
                dy = nail_y_inches - center_y_total
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    label_offset_inches = 12 / inch
                    label_x_inches = nail_x_inches + (dx / distance) * label_offset_inches
                    label_y_inches = nail_y_inches + (dy / distance) * label_offset_inches
                    
                    label_x_page = (label_x_inches - page_left_inches) * inch
                    label_y_page = (label_y_inches - page_bottom_inches) * inch
                else:
                    label_x_page = nail_x_page
                    label_y_page = nail_y_page - 15
                
                c.setFont("Helvetica-Bold", 8)
                c.setFillColor(colors.red)
                c.drawCentredString(label_x_page, label_y_page, str(nail_idx + 1))
                c.setFillColor(colors.black)
        
        # Draw circle with proper bleed
        c.setStrokeColor(colors.blue)
        c.setLineWidth(2)
        
        num_segments = 720
        for i in range(num_segments):
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi
            
            x1_inches = center_x_total + radius_inches * math.cos(angle1)
            y1_inches = center_y_total + radius_inches * math.sin(angle1)
            x2_inches = center_x_total + radius_inches * math.cos(angle2)
            y2_inches = center_y_total + radius_inches * math.sin(angle2)
            
            # Allow bleed
            bleed_margin = 0.3
            in_x = ((page_left_inches - bleed_margin <= x1_inches <= page_right_inches + bleed_margin) or
                   (page_left_inches - bleed_margin <= x2_inches <= page_right_inches + bleed_margin))
            in_y = ((page_bottom_inches - bleed_margin <= y1_inches <= page_top_inches + bleed_margin) or
                   (page_bottom_inches - bleed_margin <= y2_inches <= page_top_inches + bleed_margin))
            
            if in_x and in_y:
                x1_page = (x1_inches - page_left_inches) * inch
                y1_page = (y1_inches - page_bottom_inches) * inch
                x2_page = (x2_inches - page_left_inches) * inch
                y2_page = (y2_inches - page_bottom_inches) * inch
                
                c.line(x1_page, y1_page, x2_page, y2_page)
        
        # Footer
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.grey)
        c.drawString(10, 10, f"Radius: {radius_cm}cm | Nails: {num_nails} | PRINT AT 100% SCALE")
        c.setFillColor(colors.black)
        
        c.showPage()
    
    c.save()
    buffer.seek(0)
    
    print(f"✅ PDF generated with {len(pages_with_content)} pages!")
    
    return buffer

# =============================================================================
# Background Pre-Processing Function
# =============================================================================
def preprocess_image_background(image_bytes, num_nails, image_resolution, sid):
    """Pre-compute Radon transform and interpolation in background"""
    global preprocessing_cache, preprocessing_in_progress, cancel_flags
    
    try:
        preprocessing_in_progress[sid] = True
        
        print(f"\n📄 Starting background pre-processing for session {sid[:8]}...")
        socketio.emit('status', {'msg': '⚙️ Pre-processing started...'}, to=sid)
        socketio.sleep(0.01)  # Allow status to be sent
        
        # Check for cancellation at start
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled for session {sid[:8]} (before start)")
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        t_total_start = time.time()
        
        d, p_min, tstart, tend = 0.036, 0.00016, 0.0014, 0.0161
        R = 1.0
        ideal_image_size = int(image_resolution)
        Num_Nails = int(num_nails)
        
        # Image loading
        print("📸 Loading and processing image...")
        socketio.emit('status', {'msg': '📸 Loading image...'}, to=sid)
        t_start = time.time()
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        BW = img.resize((ideal_image_size, ideal_image_size)).transpose(Image.FLIP_TOP_BOTTOM)
        BW_array = np.array(BW) / 255.0
        t_image = time.time() - t_start
        print(f"  ⏱️ Image loading: {t_image:.3f}s")
        
        # Check cancellation
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled after image loading")
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        # Mask creation
        print("🎭 Creating circular mask...")
        socketio.emit('status', {'msg': '🎭 Creating mask...'}, to=sid)
        t_start = time.time()
        x = np.linspace(-R, R, ideal_image_size)
        y = np.linspace(-R, R, ideal_image_size)
        X, Y = np.meshgrid(x, y)
        circular_mask = X**2 + Y**2 <= R**2
        
        BW_array[~circular_mask] = 1.0
        f = 1 - BW_array
        f[~circular_mask] = 0
        t_mask = time.time() - t_start
        print(f"  ⏱️ Mask creation: {t_mask:.3f}s")
        
        # Check cancellation BEFORE starting Radon (the slow part)
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled before Radon transform")
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        # Radon transform
        print("🔄 Computing Radon transform (this may take a moment)...")
        socketio.emit('status', {'msg': '🔄 Computing Radon transform...'}, to=sid)
        socketio.sleep(0.01)  # Allow status to be sent before long operation
        t_start = time.time()
        alpha_deg = np.linspace(0., 180., 3 * Num_Nails, endpoint=False)
        p, s = radon_fun(f, alpha_deg, R, ideal_image_size)
        t_radon = time.time() - t_start
        print(f"  ⏱️ Radon transform: {t_radon:.3f}s")
        
        # Check cancellation immediately after Radon
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled after Radon transform")
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        # Filtering
        print("🔎 Filtering radon data...")
        socketio.emit('status', {'msg': '🔎 Filtering data...'}, to=sid)
        t_start = time.time()
        ind_keep = np.abs(s) < R
        s, p = s[ind_keep], p[ind_keep, :]
        
        alpha_rad = np.deg2rad(alpha_deg)
        ALPHA, S = np.meshgrid(alpha_rad, s)
        L_alpha_s = 2 * np.sqrt(np.maximum(0, R**2 - S**2))
        p = p / (L_alpha_s + 1e-12)
        t_filter = time.time() - t_start
        print(f"  ⏱️ Filtering: {t_filter:.3f}s")
        
        # Check cancellation
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled after filtering")
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        # PSI grid creation
        print("🔧 Creating PSI grid...")
        socketio.emit('status', {'msg': '🔧 Creating PSI grid...'}, to=sid)
        t_start = time.time()
        psi_1 = np.linspace(-np.pi, np.pi, Num_Nails + 1)
        psi_2 = np.linspace(0, 2 * np.pi, Num_Nails + 1)
        PSI_1, PSI_2 = np.meshgrid(psi_1, psi_2)
        
        angle_diff = np.abs(PSI_2 - PSI_1)
        L = 2 * R * np.sin(angle_diff / 2)
        L = np.maximum(L, 1e-12)
        t_psi = time.time() - t_start
        print(f"  ⏱️ PSI grid: {t_psi:.3f}s")
        
        # Check cancellation BEFORE interpolation (another slow part)
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled before interpolation")
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        # Interpolation
        print("🔀 Interpolating to PSI coordinates (this may take a moment)...")
        socketio.emit('status', {'msg': '🔀 Interpolating data...'}, to=sid)
        socketio.sleep(0.01)  # Allow status to be sent before long operation
        t_start = time.time()
        p_interpolated = AlphaS2Phi(ALPHA, S, PSI_1, PSI_2, p, R)
        p_interpolated = p_interpolated.astype(np.float32)
        t_interp = time.time() - t_start
        print(f"  ⏱️ Interpolation: {t_interp:.3f}s")

        # ADD THESE TWO LINES:
        print(f"  📊 p stats: min={np.min(p_interpolated):.6f}, max={np.max(p_interpolated):.6f}, mean={np.mean(p_interpolated):.6f}")
        print(f"  📊 NaN count: {np.sum(np.isnan(p_interpolated))}, Zero count: {np.sum(p_interpolated == 0)}")
        
        # Check cancellation before caching
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled after interpolation")
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        # Caching - ONLY if not cancelled
        print("💾 Caching results...")
        t_start = time.time()
        cache_key = f"{num_nails}_{image_resolution}"
        
        # Double-check cancel flag right before caching
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled before caching")
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        preprocessing_cache[sid] = {
            'p': p_interpolated,
            'PSI_1': PSI_1,
            'PSI_2': PSI_2,
            'L': L,
            'psi_1': psi_1,
            'psi_2': psi_2,
            'R': R,
            'cache_key': cache_key,
            'Num_Nails': Num_Nails
        }
        t_cache = time.time() - t_start
        print(f"  ⏱️ Caching: {t_cache:.3f}s")
        
        # Final check after caching - if cancelled, remove the cache we just added
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Pre-processing cancelled after caching - removing cache")
            if sid in preprocessing_cache:
                del preprocessing_cache[sid]
            socketio.emit('status', {'msg': '❌ Pre-processing cancelled'}, to=sid)
            return
        
        t_total = time.time() - t_total_start
        print(f"\n✅ Pre-processing complete! Total time: {t_total:.3f}s")
        print(f"   Breakdown: Image={t_image:.3f}s, Mask={t_mask:.3f}s, Radon={t_radon:.3f}s, Filter={t_filter:.3f}s, PSI={t_psi:.3f}s, Interp={t_interp:.3f}s\n")
        socketio.emit('status', {'msg': '✅ Pre-processing complete!'}, to=sid)
        
    except Exception as e:
        print(f"❌ Pre-processing error for {sid[:8]}: {e}")
        traceback.print_exc()
        socketio.emit('status', {'msg': f'❌ Error: {str(e)}'}, to=sid)
    finally:
        preprocessing_in_progress[sid] = False

def run_radon_algorithm(image_bytes, params, sid):
    """Main algorithm with CUDA acceleration"""
    global cancel_flags, preprocessing_cache
    
    # Check for cancellation at the very start
    if sid in cancel_flags and cancel_flags[sid]:
        print(f"🛑 Generation cancelled before starting for {sid[:8]}")
        socketio.emit('status', {'msg': '❌ Generation cancelled'}, to=sid)
        socketio.emit('generation_stopped', {}, to=sid)
        return
    
    cancel_flags[sid] = False
    
    try:
        Num_Nails = int(params.get('num_nails', 250))
        num_max_lines = int(params.get('num_strings', 4000))
        circle_radius_cm = float(params.get('circle_radius_cm', 30))
        thread_thickness_mm = float(params.get('thread_thickness_mm', 0.5))
        ideal_image_size = int(params.get('image_resolution', 300))
        
        d, p_min, tstart, tend = 0.036, 0.00016, 0.0014, 0.0161
        p_theshold = 0.0037
        R = 1.0
        
        print(f"\n{'='*60}")
        print(f"String Art Generation")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  Nails: {Num_Nails}")
        print(f"  Max Lines: {num_max_lines}")
        print(f"  Physical: {circle_radius_cm}cm radius, {thread_thickness_mm}mm thread")
        print(f"  Resolution: {ideal_image_size}x{ideal_image_size}")
        print(f"  Acceleration: {'CUDA (Native C++)' if CUDA_AVAILABLE else 'CPU (NumPy/SciPy)'}")
        print(f"{'='*60}\n")
        
        cache_key = f"{Num_Nails}_{ideal_image_size}"
        use_cache = False
        
        if sid in preprocessing_cache:
            cached = preprocessing_cache[sid]
            if cached['cache_key'] == cache_key:
                use_cache = True
                print("🚀 Using pre-processed data from cache!")
                p = cached['p']
                PSI_1 = cached['PSI_1']
                PSI_2 = cached['PSI_2']
                L = cached['L']
                psi_1 = cached['psi_1']
                psi_2 = cached['psi_2']
                R = cached['R']
        
        # Check if cancelled after trying to use cache
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Generation cancelled after cache check for {sid[:8]}")
            socketio.emit('status', {'msg': '❌ Generation cancelled'}, to=sid)
            socketio.emit('generation_stopped', {}, to=sid)
            return
        
        if not use_cache:
            print("⚙️ Computing from scratch (no cache available)")
            print("📸 Loading image...")
            t_start = time.time()
            img = Image.open(io.BytesIO(image_bytes)).convert('L')
            BW = img.resize((ideal_image_size, ideal_image_size)).transpose(Image.FLIP_TOP_BOTTOM)
            BW_array = np.array(BW) / 255.0
            
            x = np.linspace(-R, R, ideal_image_size)
            y = np.linspace(-R, R, ideal_image_size)
            X, Y = np.meshgrid(x, y)
            circular_mask = X**2 + Y**2 <= R**2
            
            BW_array[~circular_mask] = 1.0
            f = 1 - BW_array
            f[~circular_mask] = 0
            t_image = time.time() - t_start
            print(f"  ⏱️ Image loading: {t_image:.2f}s")
            
            print("🔄 Computing Radon transform...")
            t_start = time.time()
            alpha_deg = np.linspace(0., 180., 3 * Num_Nails, endpoint=False)
            p, s = radon_fun(f, alpha_deg, R, ideal_image_size)
            
            ind_keep = np.abs(s) < R
            s, p = s[ind_keep], p[ind_keep, :]
            
            alpha_rad = np.deg2rad(alpha_deg)
            ALPHA, S = np.meshgrid(alpha_rad, s)
            L_alpha_s = 2 * np.sqrt(np.maximum(0, R**2 - S**2))
            p = p / (L_alpha_s + 1e-12)
            t_radon = time.time() - t_start
            print(f"  ⏱️ Radon transform: {t_radon:.2f}s")
            
            print("🔧 Setting up PSI grid...")
            t_start = time.time()
            psi_1 = np.linspace(-np.pi, np.pi, Num_Nails + 1)
            psi_2 = np.linspace(0, 2 * np.pi, Num_Nails + 1)
            PSI_1, PSI_2 = np.meshgrid(psi_1, psi_2)
            
            angle_diff = np.abs(PSI_2 - PSI_1)
            L = 2 * R * np.sin(angle_diff / 2)
            L = np.maximum(L, 1e-12)
            t_psi = time.time() - t_start
            print(f"  ⏱️ PSI grid setup: {t_psi:.2f}s")
            
            print("🔀 Interpolating to PSI grid...")
            t_start = time.time()
            p = AlphaS2Phi(ALPHA, S, PSI_1, PSI_2, p, R)
            p = p.astype(np.float32)
            t_interp = time.time() - t_start
            print(f"  ⏱️ Interpolation: {t_interp:.2f}s")
        
        print("🎯 Starting optimized greedy algorithm with real-time drawing...\n")
        
        # Final check before starting greedy algorithm
        if sid in cancel_flags and cancel_flags[sid]:
            print(f"🛑 Generation cancelled before greedy algorithm for {sid[:8]}")
            socketio.emit('status', {'msg': '❌ Generation cancelled'}, to=sid)
            socketio.emit('generation_stopped', {}, to=sid)
            return
        
        t_greedy_start = time.time()
        psi_10, psi_20 = [], []
        nails_used = []
        size_p = p.shape
        row, col = 0, 0
        
        p = p.astype(np.float32)
        PSI_1 = PSI_1.astype(np.float32)
        PSI_2 = PSI_2.astype(np.float32)
        L = L.astype(np.float32)
        
        psi_1_np = psi_1.astype(np.float32)
        psi_2_np = psi_2.astype(np.float32)
        
        for i in range(num_max_lines):
            matlab_i = i + 1
            
            if matlab_i % 50 == 0 and sid in cancel_flags and cancel_flags[sid]:
                print(f"\n🛑 CANCELLED at iteration {matlab_i}\n")
                socketio.emit('status', {'msg': f'❌ Cancelled after {matlab_i} lines.'}, to=sid)
                socketio.emit('generation_stopped', {}, to=sid)
                break
            
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
                
                psi_10.append(float(psi_1_np[col]))
                psi_20.append(float(psi_2_np[row]))
                current_nail = int(np.round(float(psi_2_np[row]) * Num_Nails / (2 * np.pi)))
            else:
                if CUDA_AVAILABLE:
                    p_max_val, col = radon_cuda.find_max_in_row(p, row)
                else:
                    p_max_val = float(np.nanmax(p[row, :]))
                    col = int(np.nanargmax(p[row, :]))
                
                psi_10.append(float(psi_1_np[col]))
                psi_20.append(float(psi_2_np[row]))
                psi_1_val = float(psi_1_np[col])
                current_nail = int(np.round(np.mod(psi_1_val + 2*np.pi, 2*np.pi) * Num_Nails / (2 * np.pi)))
            
            nails_used.append(current_nail)
            
            if len(nails_used) >= 2:
                socketio.emit('new_line', {'start': nails_used[-2], 'end': nails_used[-1]}, to=sid)
                if matlab_i % 50 == 0:
                    socketio.sleep(0)
            
            if p_max_val < p_theshold:
                print(f'✅ Threshold reached at iteration {matlab_i}')
                socketio.emit('status', {'msg': f'Threshold at {matlab_i} lines.'}, to=sid)
                break
            
            psi_1_current = psi_10[-1]
            psi_2_current = psi_20[-1]
            s0 = R * np.cos((psi_2_current - psi_1_current) / 2)
            alpha0 = (psi_1_current + psi_2_current) / 2
            
            p_line_theory = p_line_fun(alpha0, s0, PSI_1, PSI_2, R, L, tstart, tend, d, p_min)
            
            if CUDA_AVAILABLE:
                radon_cuda.subtract_p_line(p, p_line_theory)
            else:
                p = p - p_line_theory
                p = np.maximum(p, -0.1)
            
            if matlab_i >= 20 and matlab_i % 20 == 0:
                recent = nails_used[-20:]
                if len(set(recent)) <= 3:
                    print(f"⚠️ WARNING: Algorithm stuck in loop at iteration {matlab_i}")
                    socketio.emit('status', {'msg': f'Error: stuck at iteration {matlab_i}'}, to=sid)
                    break
            
            if matlab_i % 200 == 0:
                progress = (matlab_i / num_max_lines) * 100
                socketio.emit('progress', {'percent': progress}, to=sid)
                socketio.sleep(0.001)
            
            if matlab_i % 1000 == 0:
                print(f"  Progress: {matlab_i}/{num_max_lines} lines, p_max={p_max_val:.6f}")
        
        t_greedy = time.time() - t_greedy_start
        print(f"  ⏱️ Greedy algorithm: {t_greedy:.2f}s")
        
        print("📊 Generating final output...")
        t_start = time.time()
        num_lines = len(psi_10)
        print(f'\n✅ Lines generated: {num_lines}')
        
        if num_lines > 0:
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
            
            socketio.emit('final_sequence', {
                'sequence': List_1based,
                'physical_info': {
                    'radius_cm': circle_radius_cm,
                    'radius_m': circle_radius_cm / 100.0,
                    'thread_mm': thread_thickness_mm,
                    'num_nails': Num_Nails,
                    'num_lines': num_lines,
                    'total_length_m': total_length
                }
            }, to=sid)
            
            socketio.emit('status', {'msg': f'Complete! {num_lines} lines.'}, to=sid)
            
            t_output = time.time() - t_start
            print(f"  ⏱️ Output generation: {t_output:.2f}s")
            print("✨ Done!\n")
        
    except Exception as e:
        print(f"❌ ERROR:")
        traceback.print_exc()
        socketio.emit('status', {'msg': f'Error: {str(e)}'}, to=sid)
    finally:
        if sid in cancel_flags:
            del cancel_flags[sid]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test_route():
    print("🎯 TEST ROUTE HIT!")
    return "Flask routing works!"

@app.route('/health')
def health_check():
    """Health check endpoint for Cloudflare tunnel"""
    return {
        "status": "healthy",
        "server": "Socket.IO (app_cuda.py)",
        "cuda_available": CUDA_AVAILABLE,
        "port": 8080
    }, 200

@app.route('/download_template/<num_nails>/<radius_cm>')
def download_template(num_nails, radius_cm):
    """Generate and download printable template PDF"""
    try:
        num_nails = int(num_nails)
        radius_cm = float(radius_cm)
        
        pdf_buffer = generate_printable_template(num_nails, radius_cm)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'string_art_template_{num_nails}nails_{int(radius_cm * 2)}cm.pdf'
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    global cancel_flags, preprocessing_cache, preprocessing_in_progress
    if request.sid in cancel_flags:
        del cancel_flags[request.sid]
    if request.sid in preprocessing_cache:
        del preprocessing_cache[request.sid]
    if request.sid in preprocessing_in_progress:
        del preprocessing_in_progress[request.sid]

@socketio.on('cancel_generation')
def handle_cancel():
    global cancel_flags, preprocessing_cache
    sid = request.sid
    cancel_flags[sid] = True
    
    # Also clear any cached preprocessing data for this session
    if sid in preprocessing_cache:
        print(f"🗑️ Clearing cached preprocessing data for session {sid[:8]}")
        del preprocessing_cache[sid]
    
    print(f"🛑 Cancel requested for session {sid[:8]}")
    socketio.emit('status', {'msg': '🛑 Cancelling...'}, to=sid)

@socketio.on('preprocess_image')
def handle_preprocess(data):
    global cancel_flags
    sid = request.sid
    
    # Initialize cancel flag
    cancel_flags[sid] = False
    
    image_data_url = data['imageData']
    num_nails = int(data.get('num_nails', 250))
    image_resolution = int(data.get('image_resolution', 300))
    
    header, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    
    socketio.start_background_task(preprocess_image_background, image_bytes, num_nails, image_resolution, sid)

@socketio.on('start_generation')
def handle_start_generation(data):
    global cancel_flags
    sid = request.sid
    
    # Reset cancel flag for new generation
    cancel_flags[sid] = False
    
    image_data_url = data['imageData']
    params = data['params']
    
    header, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    
    print(f"🚀 Starting generation for session {sid[:8]}")
    socketio.start_background_task(run_radon_algorithm, image_bytes, params, sid)

if __name__ == '__main__':
    port = 8080
    host = '0.0.0.0'  # Listen on all interfaces for Cloudflare tunnel access
    print(f"\n{'='*60}")
    print(f"🎨 String Art Server (Socket.IO)")
    print(f"{'='*60}")
    if CUDA_AVAILABLE:
        print("✅ CUDA Module: Loaded")
    else:
        print("⚠️ CUDA Module: Not loaded")
    print(f"\n🌐 Server: http://{host}:{port}")
    print(f"   Local: http://127.0.0.1:{port}")
    print(f"   Cloudflare: https://home-gpu.stringartapp.com")
    print(f"{'='*60}\n")

    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)