# Implementation of Quantization after ANN-to-SNN conversion
This project is an example where we train an instance of an ANN, convert it to an SNN (Spiking Neural Network) and apply channel-wise linear quantization to it. Python has been used here.
## How to install this project
1. Clone this project
2. To create an instance of a given network architecture and trains it on a given dataset, use the following command
   
<br /> ```python main_train.py --epochs=300 -dev=0 -L=4 -data=cifar10``` <br/>
3. To convert the Artifical Neural Network to an SNN and quantize it, use the following command.
 <br/>``` python main_test.py -id=vgg16_L[8] -data=cifar10 -T=8 -dev=0```<br/>
5. The accuracy of the quantized architecture is the output.
6. If you want to check the accuray of the SNN before quantization to compare it with the after quantization accuarcy, comment the quantization code in the main_test.py file and then run the command to convert ANN to SNN.

Special thanks to Arun Bhat ## Links (https://github.com/arunkone07) and the code from the research paper "Optimal ANN-SNN conversion for high-accuracy and ultra-low-latency spiking neural networks" who's code acted as the base for mine.
