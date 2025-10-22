#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

// External C functions from CUDA
extern "C" {
    void radon_transform_cuda(
        const float* h_image,
        float* h_sinogram,
        int img_size,
        int num_angles,
        const float* h_angles_deg,
        float R
    );
    
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
    );
    
    void p_line_cuda(
        float alpha0, float s0,
        const float* h_PSI_1,
        const float* h_PSI_2,
        const float* h_L,
        float* h_p_line,
        int rows, int cols,
        float R, float tstart, float tend,
        float d, float p_min
    );
    
    void subtract_p_line_cuda(
        float* h_p,
        const float* h_p_line,
        int size
    );
}

// ============================================================================
// Python-callable Radon Transform
// ============================================================================
py::array_t<float> radon_transform(
    py::array_t<float> image,
    py::array_t<float> angles_deg,
    float R
) {
    auto img_buf = image.request();
    auto angle_buf = angles_deg.request();
    
    if (img_buf.ndim != 2) {
        throw std::runtime_error("Image must be 2D");
    }
    
    int img_size = img_buf.shape[0];
    int num_angles = angle_buf.size;
    
    // Allocate output
    py::array_t<float> sinogram({img_size, num_angles});
    auto sino_buf = sinogram.request();
    
    // Call CUDA function
    radon_transform_cuda(
        static_cast<float*>(img_buf.ptr),
        static_cast<float*>(sino_buf.ptr),
        img_size,
        num_angles,
        static_cast<float*>(angle_buf.ptr),
        R
    );
    
    return sinogram;
}

// ============================================================================
// Python-callable GPU Grid Interpolation
// ============================================================================
py::array_t<float> grid_interpolation_gpu(
    py::array_t<float> p_values,
    py::array_t<float> PSI_1,
    py::array_t<float> PSI_2,
    float R,
    float alpha_min,
    float alpha_max,
    float s_min,
    float s_max
) {
    auto p_buf = p_values.request();
    auto psi1_buf = PSI_1.request();
    auto psi2_buf = PSI_2.request();
    
    if (p_buf.ndim != 2) {
        throw std::runtime_error("p_values must be 2D");
    }
    if (psi1_buf.ndim != 2 || psi2_buf.ndim != 2) {
        throw std::runtime_error("PSI grids must be 2D");
    }
    
    int num_s = p_buf.shape[0];
    int num_angles = p_buf.shape[1];
    int psi_rows = psi1_buf.shape[0];
    int psi_cols = psi1_buf.shape[1];
    
    // Allocate output
    py::array_t<float> output({psi_rows, psi_cols});
    auto out_buf = output.request();
    
    // Call CUDA function
    grid_interpolation_cuda(
        static_cast<float*>(p_buf.ptr),
        static_cast<float*>(psi1_buf.ptr),
        static_cast<float*>(psi2_buf.ptr),
        static_cast<float*>(out_buf.ptr),
        num_angles, num_s,
        psi_rows, psi_cols,
        R, alpha_min, alpha_max, s_min, s_max
    );
    
    return output;
}

// ============================================================================
// Python-callable p_line function
// ============================================================================
py::array_t<float> calculate_p_line(
    float alpha0,
    float s0,
    py::array_t<float> PSI_1,
    py::array_t<float> PSI_2,
    py::array_t<float> L,
    float R,
    float tstart,
    float tend,
    float d,
    float p_min
) {
    auto psi1_buf = PSI_1.request();
    auto psi2_buf = PSI_2.request();
    auto l_buf = L.request();
    
    if (psi1_buf.ndim != 2 || psi2_buf.ndim != 2 || l_buf.ndim != 2) {
        throw std::runtime_error("All inputs must be 2D arrays");
    }
    
    int rows = psi1_buf.shape[0];
    int cols = psi1_buf.shape[1];
    
    // Allocate output
    py::array_t<float> p_line({rows, cols});
    auto p_line_buf = p_line.request();
    
    // Call CUDA function
    p_line_cuda(
        alpha0, s0,
        static_cast<float*>(psi1_buf.ptr),
        static_cast<float*>(psi2_buf.ptr),
        static_cast<float*>(l_buf.ptr),
        static_cast<float*>(p_line_buf.ptr),
        rows, cols,
        R, tstart, tend, d, p_min
    );
    
    return p_line;
}

// ============================================================================
// Python-callable subtract function
// ============================================================================
void subtract_p_line(
    py::array_t<float> p,
    py::array_t<float> p_line
) {
    auto p_buf = p.request();
    auto p_line_buf = p_line.request();
    
    if (p_buf.size != p_line_buf.size) {
        throw std::runtime_error("Arrays must have the same size");
    }
    
    subtract_p_line_cuda(
        static_cast<float*>(p_buf.ptr),
        static_cast<float*>(p_line_buf.ptr),
        p_buf.size
    );
}

// ============================================================================
// Python-callable max finder
// ============================================================================
py::tuple find_max_and_index(py::array_t<float> array) {
    auto buf = array.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    float max_val = -1e30f;
    int max_idx = 0;
    
    for (int i = 0; i < buf.size; i++) {
        if (!std::isnan(ptr[i]) && ptr[i] > max_val) {
            max_val = ptr[i];
            max_idx = i;
        }
    }
    
    return py::make_tuple(max_val, max_idx);
}

// ============================================================================
// Python-callable column max finder
// ============================================================================
py::tuple find_max_in_column(py::array_t<float> array, int col) {
    auto buf = array.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Array must be 2D");
    }
    
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    float* ptr = static_cast<float*>(buf.ptr);
    
    float max_val = -1e30f;
    int max_row = 0;
    
    for (int row = 0; row < rows; row++) {
        float val = ptr[row * cols + col];
        if (!std::isnan(val) && val > max_val) {
            max_val = val;
            max_row = row;
        }
    }
    
    return py::make_tuple(max_val, max_row);
}

// ============================================================================
// Python-callable row max finder
// ============================================================================
py::tuple find_max_in_row(py::array_t<float> array, int row) {
    auto buf = array.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Array must be 2D");
    }
    
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    float* ptr = static_cast<float*>(buf.ptr);
    
    float max_val = -1e30f;
    int max_col = 0;
    
    for (int col = 0; col < cols; col++) {
        float val = ptr[row * cols + col];
        if (!std::isnan(val) && val > max_val) {
            max_val = val;
            max_col = col;
        }
    }
    
    return py::make_tuple(max_val, max_col);
}

// ============================================================================
// Module Definition
// ============================================================================
PYBIND11_MODULE(radon_cuda, m) {
    m.doc() = "CUDA-accelerated Radon Transform for String Art";
    
    m.def("radon_transform", &radon_transform,
          "Compute Radon transform using CUDA",
          py::arg("image"),
          py::arg("angles_deg"),
          py::arg("R") = 1.0f);
    
    m.def("grid_interpolation_gpu", &grid_interpolation_gpu,
          "GPU-accelerated grid interpolation (replaces scipy griddata)",
          py::arg("p_values"),
          py::arg("PSI_1"),
          py::arg("PSI_2"),
          py::arg("R"),
          py::arg("alpha_min"),
          py::arg("alpha_max"),
          py::arg("s_min"),
          py::arg("s_max"));
    
    m.def("calculate_p_line", &calculate_p_line,
          "Calculate p_line using CUDA",
          py::arg("alpha0"),
          py::arg("s0"),
          py::arg("PSI_1"),
          py::arg("PSI_2"),
          py::arg("L"),
          py::arg("R"),
          py::arg("tstart"),
          py::arg("tend"),
          py::arg("d"),
          py::arg("p_min"));
    
    m.def("subtract_p_line", &subtract_p_line,
          "Subtract p_line from p using CUDA",
          py::arg("p"),
          py::arg("p_line"));
    
    m.def("find_max_and_index", &find_max_and_index,
          "Find maximum value and its index",
          py::arg("array"));
    
    m.def("find_max_in_column", &find_max_in_column,
          "Find maximum value in a column",
          py::arg("array"),
          py::arg("col"));
    
    m.def("find_max_in_row", &find_max_in_row,
          "Find maximum value in a row",
          py::arg("array"),
          py::arg("row"));
}