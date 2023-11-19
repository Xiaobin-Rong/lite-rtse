# Lite-RTSE

This is an unofficial pytorch implementation of Lite-RTSE proposed in [Lite-RTSE: Exploring A Cost-Effective Lite DNN Model for Real-Time Speech Enhancement in RTC Scenarios](https://ieeexplore.ieee.org/document/10308952).

## About Lite-RTSE
<div align=center>
<img src=./pictures/Lite-RTSE.jpg#pic_center/><br>
Fig. 1. The framework of Lite-RTSE.
</div> <br>

Lite-RTSE is a lite real-time speech enhancement (SE) model, only containing 1.56 M
parameters at 0.55 G multiply-accumulate operations per second (MAC/S). The model adopts a two-stage complex spectrum reconstruction scheme of ‘masking + residual’, 
aiming at better quality and intelligibility of enhanced speech.

<div align=center>
<img src=./pictures/Multi-order%20Conv.jpg#pic_center width=400 height=200/><br>
Fig. 2. The diagram of Multi-order Convolution Block.
</div> <br>

In my view, the contributions of Lite-RTSE are as follows:
* A novel backbone that differs from the U-Net structure and the VGG-like structure, both of which are popular and successful in SE.
* Multi-order Convolution Blocks that capture local time-frequency information through a multi-order design.
* A novel spectrum compensation mechanism for both magnitude and phase.

## Issues
There are several issues as follows:
* Does the Conv-Block in Fig. 1 correspond to the multi-scale block mentioned in the paper?
* Is there only one activation function and no batch normalization (BN) applied in the network, as implied by Fig. 1, which is unusual for DNN design?
* How can the outputs from multi-order block $\tilde{Z}$ be concatenated with $|Y|$ and $|\hat{S_1|}$, considering $\tilde{Z}$ has 64 channels while both $|Y|$ and $|\hat{S_1|}$ have only one channel?
* According to my implementation, Lite-RTSE has 1.58 M parameters and 0.95 GFLOPs. The former matches the data in the paper, but the latter does not.
* The paper suggests that DCCRN outperforms GaGNet significantly, which contradicts the results presented in the [GaGNet](https://arxiv.org/abs/2106.11789) paper.

## Experiments
### Setup
We make some modification as follows:
* change the output channels of Multi-Order Conv-Block from $C$ to $1$, so that the output can be concatenated with other two features;
* use depth-wise convolution instead of standard convolution for the 3x3 convolutional layer in Conv-Block. We observe loss divergence when using Conv-Block, and adding BN or employ depth-wise convolution can alleviate this issue. We choose the latter for its significant computational reduction.

The final version of Lite-RTSE has **1.56 M** parameters and **0.66 GFLOPs**. 

### Datasets
We evaluate Lite-RTSE using two datasets on the **VCTK-DEMAND** dataset, which contains paired clean and pre-mixed noisy speech. The training and test set consist of 11,572 utterances from 28 speakers and 872
utterances from two speakers, respectively. 1,572 utterances in training set are selected for validation. The utterances are resampled to 16 kHz.

## Results
| **Model** | **Params (M)** | **FLOPs (G)** | **SISNR** | **PESQ** | **STOI** |
|:---------:|:--------------:|:-------------:|:---------:|:--------:|:--------:|
| **DCCRN**    | 3.7         | 11.13         | -         | 2.54     | 0.938    |
| **Lite-RTSE**| 1.56        | 0.66          | **18.4**  | 2.67     | 0.936    |
| **DPCRN-CF** | **0.43**    | **0.3**       | **18.4**  | **3.18** | **0.948**|

We compare Lite-RTSE with DCCRN and [DPCRN-CF](https://arxiv.org/abs/2306.00812). The results of DCCRN are provided in [S-DCCRN](https://ieeexplore.ieee.org/abstract/document/9747029) paper, and the results of DPCRN-CF are provided in the DPCRN-CF paper. The results show that Lite-RTSE outperforms DCCRN while significantly falls behind DPCRN-CF.

## Declaration
Considering the aforementioned modifications, our implementation may differ significantly from the original Lite-RTSE. We are uncertain if our version fully aligns with the paper. Please inform us of any errors or discrepancies in our implementation.
