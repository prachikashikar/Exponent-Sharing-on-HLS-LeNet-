# Exponent Sharing in LeNet

The HLS implementation of LeNet is taken from [here](https://github.com/a2824256/HLS-LeNet). Further, it is modified to share exponents in a layerwise fashion. Any layer, in the end, does Generalized Matrix Mulplications between input and weights. Here weights are stored as proposed in [[1]](#1). The implementation of an independent Generalized Matrix Multiplication (GEMM) is shared [#here](https://github.com/prachikashikar/Expo-Share-In-GEMM).


## References
<a id="1">[1]</a> 
P. Kashikar, S. Sinha and A. K. Verma, "Exploiting Weight Statistics for Compressed Neural Network Implementation on Hardware," 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS), 2021, pp. 1-4, doi: 10.1109/AICAS51828.2021.9458581.
