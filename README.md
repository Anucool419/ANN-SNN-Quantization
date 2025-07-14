# Implementation of Quantization after ANN-to-SNN conversion
This project is an example where we train an instance of an ANN, convert it to an SNN (Spiking Neural Network) and apply channel-wise linear quantization to it.
## How to install this project
1. Clone this project
2. python main_train.py --epochs=300 -dev=0 -L=4 -data=cifar10
3. The previous command would create an instance of the given network architecture and trains it on the given dataset.
4. python main_test.py -id=vgg16_L[8] -data=cifar10 -T=8 -dev=0
5. This command has converted the ANN to SNN and quantized the SNN. The accuracy of the quantized architecture is the output.
6. If you want to check the accuray of the SNN before quantization to compare it with the after quantization accuarcy, comment the quantization code in the main_test.py file and then run the command to convert ANN to SNN.
7. 
