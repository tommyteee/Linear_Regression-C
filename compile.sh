#!/bin/bash

echo "compiling started ...."

# Add the -fopenmp flag for OpenMP support
g++ -c -fopenmp main.cpp -o main.o

# Add the -fopenmp flag for OpenMP support
g++ -c -fopenmp utils/linear_models/linear_regression.cpp -o utils/linear_models/linear_regression.o
g++ -c -fopenmp utils/linear_models/multiple.cpp -o utils/linear_models/multiple.o

# Add the -fopenmp flag for OpenMP support
g++ -c -fopenmp utils/data/data_loader.cpp -o utils/data/data_loader.o

# Add the -fopenmp flag for OpenMP support
g++ -fopenmp main.o utils/data/data_loader.o utils/linear_models/linear_regression.o utils/linear_models/multiple.o -o main 

echo "finished"
