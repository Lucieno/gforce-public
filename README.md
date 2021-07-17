# GForce
**GForce: GPU-Friendly Oblivious and Rapid Neural Network Inference**

A crypto-assisted framework leveraging GPU, additive secret share, and homomorphic encryption to protect the privacy of models and queries in inference.

The details of this project are presented in the following paper:

[GForce: GPU-Friendly Oblivious and Rapid Neural Network Inference](https://www.usenix.org/system/files/sec21fall-ng.pdf) <br>
**[Lucien K. L. Ng](https://lucieno.github.io/), [Sherman S. M. Chow](https://staff.ie.cuhk.edu.hk/~smchow/)** <br>
Usenix Security Symposium 2021


## Suggested Setup
We tested our code on 2 Google Cloud VMs located in the same region. They were running with
- Ubuntu 18.04 LTS
- Nvidia V100 GPU
- 8 Virtual Intel Xeon (Skylake) CPUs at 2GHz
- 52 GB RAM

Note that GPU is necessary. Nvidia P100 GPU is also well suited with our code.

## How to Install
1. Install [Anaconda](https://www.anaconda.com/), a Python package manager.

2. Build an Anaconda virtual environment and install required packages as follows:

       conda env create -f conda_requirement.yml

    It will create a virtual environment called `gforce`. Please activate this environment by:

       conda activate gforce
  
3. Install [SEAL-Python](https://github.com/Lucieno/SEAL-Python/tree/master) by following the instructions in its README.md. 

    Remember to activate the virtual environment `gforce` (by `conda activate gforce`) before installation.

## How to run

### Testing the Performance of Neural Networks
- Run VGG-16 on CIFAR-10

    - If you want to run both the server-side and client-side code in a local machine, run the following command

          python src/secure_vgg.py

    - If the server and the client are on different machines, please run:

        * The server run the following command, where `$IpClient` is the IP address of the client

              python src/secure_vgg.py --ip="$IpClient" -s 0

        * The client run the following command, where `$IpServer` is the IP address of the server

              python src/secure_vgg.py --ip="$IpServer" -s 1

- If you want to try running other neural networks (e.g., [MiniONN](https://eprint.iacr.org/2017/452)'s NN with MaxPool/AvgPool) or testing on CIFAR-100, 
    you may add the argument `--test $TestName`.
    `$TestName` can be: 
    - `vgg_cifar10`
    - `vgg_cifar100`
    - `minionn_avgpool` (for CIFAR-10)
    - `minionn_maxpool` (for CIFAR-10)

    For example, the command of running VGG-16 on CIFAR-100 on a single machine:

      python src/secure_vgg.py --test vgg_cifar100

    You may combine with the arguments `--ip="$IpClient" -s 0` and `--ip="$IpServer" -s 1` to test the cross-machine performance.

### Testing the Performance of Non-Linear Layers

You may also try testing the performance of non-linear layers.

- For ReLU Layer:

      python src/relu_dgk.py

- For Maxpool Layer (with 2x2 pooling windows):

      python src/maxpool2x2_dgk.py

The arguments python `--ip="$IpClient" -s 0` and `--ip="$IpServer" -s 1` for testing the cross-machine performance.

### Testing the Accuracy of Neural Networks
For testing the accuracy of our pre-trained GForce DNN (under modulo operations), you may run

    python src/swalp_inference.py --test $TestName

where `$TestName` can be 
- `vgg16_cifar10`
- `vgg16_cifar100`
- `minionn_avgpool` (for CIFAR-10)
- `minionn_maxpool` (for CIFAR-10)


## Disclaimer
DO NOT USE THIS SOFTWARE TO SECURE ANY REAL-WORLD DATA OR COMPUTATION!

This software is a proof-of-concept meant for performance testing of the GForce framework ONLY. It is full of security vulnerabilities that facilitate testing, debugging, and performance measurements. In any real-world deployment, these vulnerabilities can be easily exploited to leak all user inputs.

## Related Project
Our team also proposed [Goten: GPU-Outsourcing Trusted Execution of Neural Network Training](https://github.com/goten-team/Goten), a secure solution for privacy-preserving neural network training.

## Acknowledgement
Special thanks to [Anna P. Y. Woo](https://github.com/woopuiyung), who contributed a good chunk of code.
