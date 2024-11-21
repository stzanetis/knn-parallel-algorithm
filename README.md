# k-NN Parallel Algorithm
This repository contains the implementation of a parallel **k-Nearest Neighbors** algorithm that is part of the Parallel and Distributed Systems class of the Electrical and Computer Engineering department of the Aristotle University of Thessaloniki. This algorithm written in **C++** uses the **OpenBLAS** library for fast matrix calculations as well as **OpenCILK**, **OpenMP** and **Pthreads** for parallelization. The results of this implementation are later compared to Matlab's **knnsearch** function.

## Introduction
The K-Nearest Neighbors **(k-NN)** algorithm is a simple, yet powerful, classification method. This project aims to optimize this algorithm by **parallelizing** its computations using various libraries and tools.

## Installation
To get started, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/stzanetis/knn-parallel-algorithm.git
cd knn-parallel-algorithm
```
Ensure you have the following installed:

 - OpenBLAS
 - OpenCILK
 - HDF5
 - Matlab
 - GCC Compiler

## Usage
In order to be able to test this implementation you will have to create a **test** folder in the **knn-parallel-algorithm** directory where the folders **V1** and **V0** folders are also located, and store the necessary **.hdf5** files there.
After running the code you will be asked if you want to either:

 - Import matrices from a .hdf5 file.
 - Use randomly generated matrices.
 - Use some small matrices for testing and printing.
 
 After this, you will have to provide the number of the k nearest neighbors, and lastly input the name of the .hdf5 file you want to use (if you select the first option). Make sure you also add the file extension to the file name, for example: `mnist-784-euclidean.hdf5`.

## Using the provided Matlab script for performance testing
If you want to use the provided `calculatePercision.m` Matlab script you will have to make sure you have **Matlab** installed.
Each version of the **C++** code will have a different name for the **results.h5** file. For example: 
`omp-results.h5` for the **OpenMP** implementation.
which you will have to change in the first few lines of the Matlab script.
```Matlab
myidx = h5read("omp-results.h5", '/idx');
myidx = myidx';
mydist = h5read("omp-results.h5", '/dist');
mydist = mydist';
```

## Example Benchmark
The testing of the parallel algorithm was done using the datasets from the `mnist-784-euclidean.hdf5` file found on the [ANN Benchamarks](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file), with a system consisting of a **6-Core AMD 4650U** Processor and **12GB** Ram running on **WSL Ubuntu**.
