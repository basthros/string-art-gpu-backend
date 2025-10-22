from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import pybind11

class CUDAExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)

class BuildExtension(build_ext):
    def build_extensions(self):
        # Find CUDA toolkit
        cuda_path = os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0')
        
        if not os.path.exists(cuda_path):
            print(f"ERROR: CUDA not found at {cuda_path}")
            print("Please set CUDA_PATH environment variable")
            sys.exit(1)
        
        print(f"Using CUDA from: {cuda_path}")
        
        # NVCC compiler settings
        nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe')
        
        for ext in self.extensions:
            # Separate CUDA and C++ sources
            cuda_sources = [s for s in ext.sources if s.endswith('.cu')]
            cpp_sources = [s for s in ext.sources if s.endswith('.cpp')]
            
            # Compile CUDA files
            cuda_objects = []
            for cuda_file in cuda_sources:
                obj_file = cuda_file.replace('.cu', '.obj')
                print(f"Compiling {cuda_file}...")
                
                nvcc_cmd = [
                    nvcc_path,
                    '-c',
                    cuda_file,
                    '-o', obj_file,
                    '--use_fast_math',
                    '-O3',
                    '-std=c++14',
                    '-Xcompiler', '/MD',  # Match Python's runtime library
                    '--expt-relaxed-constexpr',
                    '-gencode', 'arch=compute_75,code=sm_75',  # Adjust for your GPU
                    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 30xx
                    '-gencode', 'arch=compute_89,code=sm_89',  # RTX 40xx
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
            
            # Add CUDA libraries
            ext.library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
            ext.libraries.extend(['cudart'])
            
            # Add include directories
            ext.include_dirs.append(os.path.join(cuda_path, 'include'))
            ext.include_dirs.append(pybind11.get_include())
            ext.include_dirs.append(pybind11.get_include(user=True))
        
        # Now build C++ wrapper with MSVC
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
            ],
            language='c++',
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires='>=3.7',
)