#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CUDA Kernel: Radon Transform
// ============================================================================
__global__ void radon_transform_kernel(
    const float* image,
    float* sinogram,
    int img_size,
    int num_angles,
    const float* cos_angles,
    const float* sin_angles,
    float R
) {
    int angle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (angle_idx >= num_angles) return;
    
    int sinogram_height = img_size;
    if (s_idx >= sinogram_height) return;
    
    float s = ((float)s_idx - sinogram_height / 2.0f) * (2.0f * R / img_size);
    
    float cos_a = cos_angles[angle_idx];
    float sin_a = sin_angles[angle_idx];
    
    float sum = 0.0f;
    int count = 0;
    
    // Integrate along the line perpendicular to angle
    for (int t = -img_size/2; t < img_size/2; t++) {
        float t_normalized = t * (2.0f * R / img_size);
        
        float x = s * cos_a - t_normalized * sin_a;
        float y = s * sin_a + t_normalized * cos_a;
        
        // Convert to image coordinates
        float img_x = (x / R + 1.0f) * img_size / 2.0f;
        float img_y = (y / R + 1.0f) * img_size / 2.0f;
        
        // Flip Y axis to match image coordinates (top-left origin)
        img_y = img_size - 1 - img_y;
        
        // Bilinear interpolation
        if (img_x >= 0 && img_x < img_size - 1 && img_y >= 0 && img_y < img_size - 1) {
            int x0 = (int)img_x;
            int y0 = (int)img_y;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float dx = img_x - x0;
            float dy = img_y - y0;
            
            float val = (1 - dx) * (1 - dy) * image[y0 * img_size + x0] +
                        dx * (1 - dy) * image[y0 * img_size + x1] +
                        (1 - dx) * dy * image[y1 * img_size + x0] +
                        dx * dy * image[y1 * img_size + x1];
            
            sum += val;
            count++;
        }
    }
    
    sinogram[s_idx * num_angles + angle_idx] = (count > 0) ? sum : 0.0f;
}

// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// CUDA Kernel: Grid Interpolation - GEOMETRIC VALIDATION
// ============================================================================
__global__ void grid_interpolation_kernel(
    const float* alpha_flat,      // Not used, kept for signature compatibility
    const float* s_flat,          // Not used, kept for signature compatibility
    const float* p_values,        // [num_s, num_angles] radon transform data
    const float* psi1_grid,       // [psi_rows, psi_cols] output grid
    const float* psi2_grid,       // [psi_rows, psi_cols] output grid
    float* output,                // [psi_rows, psi_cols] result
    int num_angles,               // columns in p_values
    int num_s,                    // rows in p_values
    int psi_rows,                 // output rows
    int psi_cols,                 // output cols
    float R,
    float alpha_min,              // minimum alpha in radians
    float alpha_max,              // maximum alpha in radians
    float s_min,                  // minimum s value
    float s_max                   // maximum s value
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= psi_rows || col >= psi_cols) return;
    
    int idx = row * psi_cols + col;
    float psi1 = psi1_grid[idx];
    float psi2 = psi2_grid[idx];
    
    // Transform (psi1, psi2) -> (alpha, s)
    float alpha_target = (psi1 + psi2) / 2.0f;
    float angle_diff = (psi2 - psi1) / 2.0f;
    
    // CRITICAL VALIDATION #1: angle_diff must be a valid arccos output [0, pi]
    // If not, this point is geometrically impossible
    if (angle_diff < -0.0001f || angle_diff > M_PI + 0.0001f) {
        output[idx] = 0.0f;
        return;
    }
    
    // Clamp to valid arccos range for numerical stability
    angle_diff = fmaxf(0.0f, fminf(M_PI, angle_diff));
    
    float s_over_R = cosf(angle_diff);
    float s_target = R * s_over_R;
    
    // CRITICAL VALIDATION #2: s must be physically valid
    // The radon transform only has data for |s| < R
    // Be conservative: require |s| < 0.98 * R
    if (fabsf(s_target) > 0.98f * R) {
        output[idx] = 0.0f;
        return;
    }
    
    // Normalize alpha to [0, pi] for radon periodicity
    while (alpha_target < 0.0f) alpha_target += M_PI;
    while (alpha_target >= M_PI) alpha_target -= M_PI;
    
    // CRITICAL VALIDATION #3: alpha must be in data range with strict bounds
    // Use tighter tolerance to match scipy behavior
    float alpha_tolerance = 0.02f * (alpha_max - alpha_min);
    if (alpha_target < alpha_min + alpha_tolerance || 
        alpha_target > alpha_max - alpha_tolerance) {
        output[idx] = 0.0f;
        return;
    }
    
    // CRITICAL VALIDATION #4: s must be in data range with strict bounds
    float s_tolerance = 0.02f * (s_max - s_min);
    if (s_target < s_min + s_tolerance || 
        s_target > s_max - s_tolerance) {
        output[idx] = 0.0f;
        return;
    }
    
    // CRITICAL VALIDATION #5: Verify inverse transformation consistency
    // If we transform back, we should get close to original psi values
    float psi1_check = alpha_target - angle_diff;
    float psi2_check = alpha_target + angle_diff;
    
    float psi1_error = fabsf(psi1_check - psi1);
    float psi2_error = fabsf(psi2_check - psi2);
    
    // Account for periodic wrapping in psi1 (which can be negative)
    if (psi1_error > M_PI) psi1_error = fabsf(psi1_error - 2.0f * M_PI);
    if (psi2_error > M_PI) psi2_error = fabsf(psi2_error - 2.0f * M_PI);
    
    if (psi1_error > 0.01f || psi2_error > 0.01f) {
        output[idx] = 0.0f;
        return;
    }
    
    // Map to array indices
    float alpha_frac = (alpha_target - alpha_min) / (alpha_max - alpha_min);
    float s_frac = (s_target - s_min) / (s_max - s_min);
    
    // Clamp to [0, 1]
    alpha_frac = fmaxf(0.0f, fminf(1.0f, alpha_frac));
    s_frac = fmaxf(0.0f, fminf(1.0f, s_frac));
    
    float alpha_idx_f = alpha_frac * (float)(num_angles - 1);
    float s_idx_f = s_frac * (float)(num_s - 1);
    
    // Get integer indices
    int alpha_i0 = (int)floorf(alpha_idx_f);
    int alpha_i1 = min(alpha_i0 + 1, num_angles - 1);
    int s_i0 = (int)floorf(s_idx_f);
    int s_i1 = min(s_i0 + 1, num_s - 1);
    
    // Interpolation weights
    float alpha_w = alpha_idx_f - (float)alpha_i0;
    float s_w = s_idx_f - (float)s_i0;
    
    // Fetch values
    float p00 = p_values[s_i0 * num_angles + alpha_i0];
    float p01 = p_values[s_i0 * num_angles + alpha_i1];
    float p10 = p_values[s_i1 * num_angles + alpha_i0];
    float p11 = p_values[s_i1 * num_angles + alpha_i1];
    
    // Check for invalid source data
    if (!isfinite(p00) || !isfinite(p01) || !isfinite(p10) || !isfinite(p11)) {
        output[idx] = 0.0f;
        return;
    }
    
    // Bilinear interpolation
    float p0 = p00 * (1.0f - alpha_w) + p01 * alpha_w;
    float p1 = p10 * (1.0f - alpha_w) + p11 * alpha_w;
    float result = p0 * (1.0f - s_w) + p1 * s_w;
    
    output[idx] = isfinite(result) ? result : 0.0f;
}
// ============================================================================
// CUDA Kernel: p_region function
// ============================================================================
__device__ float p_region_device(
    float alpha0, float s0,
    float ALPHA, float S,
    float R, float t
) {
    float sin_term = sinf(ALPHA - alpha0);
    float numerator = S * S + s0 * s0 - 2.0f * S * s0 * cosf(ALPHA - alpha0);
    float denominator = sin_term * sin_term + (t / (2.0f * R)) * (t / (2.0f * R));
    denominator = fmaxf(denominator, 1e-12f);
    return numerator / denominator - R * R;
}

// ============================================================================
// CUDA Kernel: mask function
// ============================================================================
__global__ void mask_kernel(
    float alpha0, float s0,
    const float* ALPHA,
    const float* S,
    float* mask,
    int size,
    float R,
    float tstart,
    float tend
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float alpha_val = ALPHA[idx];
    float s_val = S[idx];
    
    // Initialize mask
    float region_start = p_region_device(alpha0, s0, alpha_val, s_val, R, tstart);
    mask[idx] = (region_start < 0.0f) ? 1.0f : 0.0f;
    
    // Fade region
    const int n = 4;
    float t_step = (tend - tstart) / (n - 1);
    
    for (int i = 0; i < n - 1; i++) {
        float t_i = tstart + i * t_step;
        float t_i_plus_1 = tstart + (i + 1) * t_step;
        
        float region_i = p_region_device(alpha0, s0, alpha_val, s_val, R, t_i);
        float region_i_plus_1 = p_region_device(alpha0, s0, alpha_val, s_val, R, t_i_plus_1);
        
        if (region_i > 0.0f && region_i_plus_1 < 0.0f) {
            if (tend > tstart) {
                mask[idx] = (tend - t_i) / (tend - tstart);
            } else {
                mask[idx] = 1.0f;
            }
        }
    }
}

// ============================================================================
// CUDA Kernel: p_line function (optimized)
// ============================================================================
__global__ void p_line_kernel(
    float alpha0, float s0,
    const float* PSI_1,
    const float* PSI_2,
    const float* L,
    float* p_line,
    int size,
    float R,
    float tstart,
    float tend,
    float d,
    float p_min
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float psi1 = PSI_1[idx];
    float psi2 = PSI_2[idx];
    float length = L[idx];
    
    float ALPHA = (psi1 + psi2) / 2.0f;
    float S = R * cosf((psi2 - psi1) / 2.0f);
    
    // Calculate mask
    float mask_val = 0.0f;
    float region_start = p_region_device(alpha0, s0, ALPHA, S, R, tstart);
    if (region_start < 0.0f) mask_val = 1.0f;
    
    // Valid line check
    float min_L = 0.01f * R;
    float valid_line = (length > min_L) ? 1.0f : 0.0f;
    
    // Calculate p_line
    float sin_term = sinf(ALPHA - alpha0);
    float sin_term_abs = fabsf(sin_term);
    
    float denominator = (d * length - p_min) * sin_term_abs + p_min;
    denominator = fmaxf(denominator, p_min / 100.0f);
    
    float result = d * p_min / denominator;
    result = result * mask_val * valid_line;
    result = fminf(result, d * 2.0f);
    
    // Handle non-finite values
    if (!isfinite(result)) result = 0.0f;
    result = fmaxf(result, 0.0f);
    
    p_line[idx] = result;
}

// ============================================================================
// CUDA Kernel: Subtract p_line from p
// ============================================================================
__global__ void subtract_kernel(
    float* p,
    const float* p_line,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    p[idx] = fmaxf(p[idx] - p_line[idx], -0.1f);
}

// ============================================================================
// CUDA Kernel: Find Maximum in Column
// ============================================================================
__global__ void find_max_column_kernel(
    const float* p,
    float* max_val,
    int* max_row,
    int rows,
    int col
) {
    extern __shared__ float shared_max[];
    extern __shared__ int shared_idx[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = -1e30f;
    int local_idx = 0;
    
    if (idx < rows) {
        float val = p[idx * gridDim.y + col];
        if (!isnan(val)) {
            local_max = val;
            local_idx = idx;
        }
    }
    
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((int*)max_val, __float_as_int(shared_max[0]));
        if (__float_as_int(shared_max[0]) == __float_as_int(*max_val)) {
            *max_row = shared_idx[0];
        }
    }
}

// ============================================================================
// Host Functions (called from Python)
// ============================================================================

extern "C" {

void radon_transform_cuda(
    const float* h_image,
    float* h_sinogram,
    int img_size,
    int num_angles,
    const float* h_angles_deg,
    float R
) {
    // Allocate device memory
    float *d_image, *d_sinogram, *d_cos_angles, *d_sin_angles;
    
    size_t img_bytes = img_size * img_size * sizeof(float);
    size_t sino_bytes = img_size * num_angles * sizeof(float);
    size_t angle_bytes = num_angles * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_image, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_sinogram, sino_bytes));
    CUDA_CHECK(cudaMalloc(&d_cos_angles, angle_bytes));
    CUDA_CHECK(cudaMalloc(&d_sin_angles, angle_bytes));
    
    // Precompute sin/cos on CPU
    float* h_cos_angles = new float[num_angles];
    float* h_sin_angles = new float[num_angles];
    for (int i = 0; i < num_angles; i++) {
        float rad = h_angles_deg[i] * M_PI / 180.0f;
        h_cos_angles[i] = cosf(rad);
        h_sin_angles[i] = sinf(rad);
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_image, h_image, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos_angles, h_cos_angles, angle_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin_angles, h_sin_angles, angle_bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((num_angles + block.x - 1) / block.x, (img_size + block.y - 1) / block.y);
    
    radon_transform_kernel<<<grid, block>>>(
        d_image, d_sinogram, img_size, num_angles,
        d_cos_angles, d_sin_angles, R
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_sinogram, d_sinogram, sino_bytes, cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_image);
    cudaFree(d_sinogram);
    cudaFree(d_cos_angles);
    cudaFree(d_sin_angles);
    delete[] h_cos_angles;
    delete[] h_sin_angles;
}

void grid_interpolation_cuda(
    const float* h_p_values,
    const float* h_psi1_grid,
    const float* h_psi2_grid,
    float* h_output,
    int num_angles,
    int num_s,
    int psi_rows,
    int psi_cols,
    float R,
    float alpha_min,
    float alpha_max,
    float s_min,
    float s_max
) {
    float *d_p_values, *d_psi1_grid, *d_psi2_grid, *d_output;
    
    size_t p_bytes = num_s * num_angles * sizeof(float);
    size_t grid_bytes = psi_rows * psi_cols * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_p_values, p_bytes));
    CUDA_CHECK(cudaMalloc(&d_psi1_grid, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_psi2_grid, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, grid_bytes));
    
    CUDA_CHECK(cudaMemcpy(d_p_values, h_p_values, p_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_psi1_grid, h_psi1_grid, grid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_psi2_grid, h_psi2_grid, grid_bytes, cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid_dim((psi_cols + block.x - 1) / block.x, (psi_rows + block.y - 1) / block.y);
    
    grid_interpolation_kernel<<<grid_dim, block>>>(
        nullptr, nullptr,
        d_p_values, d_psi1_grid, d_psi2_grid, d_output,
        num_angles, num_s, psi_rows, psi_cols,
        R, alpha_min, alpha_max, s_min, s_max
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, grid_bytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_p_values);
    cudaFree(d_psi1_grid);
    cudaFree(d_psi2_grid);
    cudaFree(d_output);
}

void p_line_cuda(
    float alpha0, float s0,
    const float* h_PSI_1,
    const float* h_PSI_2,
    const float* h_L,
    float* h_p_line,
    int rows, int cols,
    float R, float tstart, float tend,
    float d, float p_min
) {
    int size = rows * cols;
    
    float *d_PSI_1, *d_PSI_2, *d_L, *d_p_line;
    size_t bytes = size * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_PSI_1, bytes));
    CUDA_CHECK(cudaMalloc(&d_PSI_2, bytes));
    CUDA_CHECK(cudaMalloc(&d_L, bytes));
    CUDA_CHECK(cudaMalloc(&d_p_line, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_PSI_1, h_PSI_1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_PSI_2, h_PSI_2, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_L, h_L, bytes, cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    p_line_kernel<<<grid_size, block_size>>>(
        alpha0, s0, d_PSI_1, d_PSI_2, d_L, d_p_line,
        size, R, tstart, tend, d, p_min
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_p_line, d_p_line, bytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_PSI_1);
    cudaFree(d_PSI_2);
    cudaFree(d_L);
    cudaFree(d_p_line);
}

void subtract_p_line_cuda(
    float* h_p,
    const float* h_p_line,
    int size
) {
    float *d_p, *d_p_line;
    size_t bytes = size * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_p, bytes));
    CUDA_CHECK(cudaMalloc(&d_p_line, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_p, h_p, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_line, h_p_line, bytes, cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    subtract_kernel<<<grid_size, block_size>>>(d_p, d_p_line, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost));
    
    cudaFree(d_p);
    cudaFree(d_p_line);
}

} // extern "C"