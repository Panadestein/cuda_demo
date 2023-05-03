# CUDA Fortran Minimal Working Example

This repository contains a minimal working example (MWE) demonstrating how to use CUDA with Fortran using the ISO_C_BINDING module. The project uses CMake as the build system.

## Prerequisites

- CMake 3.8 or higher
- GNU compilers (gfortran, g++)
- CUDA
- BLAS/LAPACK

## Building the project

1. Clone the repository:

```bash
git clone https://github.com/Panadestein/cuda_demo.git
cd cuda_demo
```
2. Create a build directory and navigate to it:

```bash
mkdir build
cd build
```

3. Configure the project using CMake and compile:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

Note: the default build type is `Debug`.

## Contributing

Feel free to open an issue or submit a pull request if you have any suggestions, improvements, or bug fixes.
