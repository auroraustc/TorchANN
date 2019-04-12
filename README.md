# Torch-NNMD
Use torch to train NN potential

## Request:
1. pyTorch
2. C compiler (icc 2018 tested)

## Input data:
```bash
box.raw type.raw coord.raw force.raw energy.raw
```

## STEP 1: Make c executable for data pre-processing
```bash
cd ./Torch-NNMD/c
#(Adapt the makefile for your computer)
make
```

## STEP 2: Pre-process input data
```bash
cd ../test_2 #The input data for test is under this directory
../c/a.out > log
```

## STEP 3: Run python script to train
```bash
python ../python/train.py
```
