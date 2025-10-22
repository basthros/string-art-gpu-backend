from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import pybind11
import shutil # Needed for shutil.which

class CUDAExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)

class BuildExtension(build_ext):
    def build_extensions(self):
        # Find CUDA toolkit - Updated for Linux/Docker
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if not cuda_home:
            # Check standard Linux path if env var not set
            if os.path.exists('/usr/local/cuda'):
                cuda_home = '/usr/local/cuda'
            else:
                print("ERROR: CUDA toolkit path not found.")
                print("Please set CUDA_HOME or CUDA_PATH environment variable, or ensure CUDA is in /usr/local/cuda.")
                sys.exit(1)

        print(f"Using CUDA from: {cuda_home}")

        # NVCC compiler path - assumes Linux structure
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')

        # --- Check if nvcc exists ---
        if not os.path.exists(nvcc_path):
            print(f"ERROR: nvcc compiler not found at {nvcc_path}")
            # Try finding nvcc on the system PATH as a fallback
            nvcc_from_path = shutil.which('nvcc')
            if nvcc_from_path:
                print(f"Found nvcc on PATH: {nvcc_from_path}. Using that instead.")
                nvcc_path = nvcc_from_path
            else:
                print("nvcc not found on PATH either. Exiting.")
                sys.exit(1)

        # CUDA library path - assumes Linux structure
        cuda_lib_dir = os.path.join(cuda_home, 'lib64')
        if not os.path.exists(cuda_lib_dir):
            # Fallback for some installations that might use lib instead of lib64
            cuda_lib_dir_alt = os.path.join(cuda_home, 'lib')
            if os.path.exists(cuda_lib_dir_alt):
                cuda_lib_dir = cuda_lib_dir_alt
            else:
                print(f"ERROR: CUDA library directory not found at {cuda_lib_dir} or {cuda_lib_dir_alt}")
                sys.exit(1)

        print(f"Using CUDA library directory: {cuda_lib_dir}")

        for ext in self.extensions:
            # Separate CUDA and C++ sources
            cuda_sources = [s for s in ext.sources if s.endswith('.cu')]
            cpp_sources = [s for s in ext.sources if s.endswith('.cpp')]

            # Compile CUDA files
            cuda_objects = []
            for cuda_file in cuda_sources:
                # Use .o for Linux object files
                obj_file = cuda_file.replace('.cu', '.o')
                print(f"Compiling {cuda_file}...")

                nvcc_cmd = [
                    nvcc_path, # Use the path found earlier
                    '-c',
                    cuda_file,
                    '-o', obj_file,
                    '--use_fast_math',
                    '-O3',
                    '-std=c++14',
                    # Removed Windows-specific flag: '-Xcompiler', '/MD',
                    '--expt-relaxed-constexpr',
                    '-Xcompiler', '-fPIC', # Add Position Independent Code flag for shared libraries
                    '-gencode', 'arch=compute_75,code=sm_75', # Keep architectures
                    '-gencode', 'arch=compute_86,code=sm_86',
                    '-gencode', 'arch=compute_89,code=sm_89',
                ]

                print(f"\n{'='*60}")
                print("NVCC Command:")
                print(' '.join(nvcc_cmd))
                print(f"{'='*60}\n")

                result = subprocess.run(nvcc_cmd, capture_output=True, text=True)

                print("NVCC STDOUT:")
                print(result.stdout if result.stdout else "(empty)")
                print("\nNVCC STDERR:")
                print(result.stderr if result.stderr else "(empty)")
                print(f"\nReturn Code: {result.returncode}")

                if result.returncode != 0:
                    print(f"\n❌ NVCC FAILED!")
                    print(f"Command that failed:\n{' '.join(nvcc_cmd)}")
                    sys.exit(1)

                cuda_objects.append(obj_file)

            # Add CUDA object files to extra objects
            ext.extra_objects = cuda_objects

            # Remove .cu files from sources (already compiled manually)
            ext.sources = cpp_sources

            # Add CUDA libraries (Use the cuda_lib_dir found earlier)
            ext.library_dirs.append(cuda_lib_dir)
            ext.libraries.extend(['cudart'])

            # Add include directories
            ext.include_dirs.append(os.path.join(cuda_home, 'include'))
            ext.include_dirs.append(pybind11.get_include())
            ext.include_dirs.append(pybind11.get_include(user=True))

        # Now build C++ wrapper (let setuptools handle it for Linux)
        super().build_extensions()

# Setup configuration
setup(
    name='radon_cuda',
    version='1.0.0',
    author='String Art Generator',
    description='CUDA-accelerated Radon Transform',
    ext_modules=[
        CUDAExtension(
            'radon_cuda',
            sources=[
                'radon_cuda.cu',
                'radon_cuda_wrapper.cpp',
            ],
            include_dirs=[
                pybind11.get_include(),
                pybind11.get_include(user=True),
                # Add numpy includes if needed, though pybind11 often handles this
                # os.path.join(sys.prefix, 'include'),
                # os.path.join(sys.prefix, 'Lib', 'site-packages', 'numpy', 'core', 'include'),
            ],
            language='c++',
            extra_compile_args=['-std=c++14', '-fPIC'], # Common flags for Linux C++ compilation
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires='>=3.7',
)