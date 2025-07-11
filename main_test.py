import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from Models import modelpool
from Preprocess import datapool
from utils import train, val, seed_all, get_logger
from Models.layer import *

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=200, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg16',type=str,help='model')
parser.add_argument('-id', '--identifier', type=str,help='model statedict identifier')

# test configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    seed_all(args.seed)
    # preparing data
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    # preparing model
    model = modelpool(args.model, args.dataset)

    model_dir = '%s-checkpoints'% (args.dataset)
    state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
    
    # if old version state_dict
    keys = list(state_dict.keys())
    for k in keys:
        if "relu.up" in k:
            state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
        elif "up" in k:
            state_dict[k[:-2]+'thresh'] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    # for name, param in model.named_parameters():
    #     print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
        

    print("\n Analyzing modules for quantization...\n")
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            print(f"[QUANTIZABLE] {name}: {type(module)} → weight shape: {module.weight.shape}")
        elif "IF" in str(type(module)) or "Spike" in str(type(module)) or "Neuron" in str(type(module)):
            print(f"[SPIKING NEURON] {name}: {type(module)}")



    model.to(device)

    model.set_T(args.time)
    model.set_L(8)
    
    from main import evaluate_channel_quantization  # Adjust import if needed
    top1, top5 = evaluate_channel_quantization(model, test_loader, quant_layers=(nn.Conv2d, nn.Linear))
    print(f"Final Quantized Accuracy: Top-1 = {top1:.2f}%, Top-5 = {top5:.2f}%")


    # # ADD QUANTIZATION HERE
    # from .main import evaluate_channel_quantization
    # evaluate_channel_quantization(model, test_loader, quant_layers=(nn.Conv2d, nn.Linear))


    # for m in model.modules():
    #     if isinstance(m, IF):
    #         print(m.thresh)

    acc = val(model, test_loader, device, args.time)
    print(f"Validation Accuracy after quantization: {acc}")
    #print(acc)



if __name__ == "__main__":
    main()