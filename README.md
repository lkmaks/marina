# Intoduction

Lavrik-Karmazin Maksim's fork of

experiments for the paper "MARINA: Faster Non-Convex Distributed Learning with Compression" by Eduard Gorbunov, Konstantin Burlachenko, Zhize Li, Peter Richtarik. The paper is accepted for presentation and publication to Thirty-eighth International Conference on Machine Learning (ICML) 2021.

# Reference to the paper
https://arxiv.org/abs/2102.07845 - MARINA paper

# ResNet-18 @ CIFAR100 experiments (nn dir)

## Prepare environment for the experiments
```bash
conda create -n marina python=3.9.1 -y
conda activate marina
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge -y
conda install jupyter matplotlib numpy psutil -y
mkdir nn/out
```

## Description

I refactored and optimized experiments code. 
Experiments can be run via:

```
python nn_experiments_parallel.py
```

and parameters changed by configuring main() calls.

To plot results:
```
python show_all.py
```


# Experiments for binary classification with non-convex loss

## Prepare environment for the experiments
1) conda create -n marina2 python=3.8
2) conda activate marina2
3) conda install matplotlib, numpy, psutil, mpi4py
4) data/download.sh

## Description

To run experiments and plot results:

```bash
cd lin/linux/
nice -n 19 ./run.sh
```

Parameters (number of iterations KMax) can be configured in run.sh