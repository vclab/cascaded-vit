"""
Testing the speed of different models
"""
import os
import torch
import torchvision
import time
import timm
from model.build import CascadedViT_S, CascadedViT_M, CascadedViT_L, CascadedViT_XL
import torchvision
import utils
from fvcore.nn import FlopCountAnalysis

torch.autograd.set_grad_enabled(False)


T0 = 10
T1 = 60


def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

### Apple silicon
def compute_throughput_mps(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.mps.empty_cache() if hasattr(torch, "mps") else None
    start = time.time()
    while time.time() - start < T0:
        model(inputs)
        torch.mps.synchronize()
    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        torch.mps.synchronize()
        timing.append(time.time() - start)
    timing = torch.tensor(timing)
    print(name, device, batch_size/timing.mean().item(),
          'image/s @ batch size', batch_size)

for device in ['cuda:0', 'mps', 'cpu']:

    if device == 'cuda':
        if not torch.cuda.is_available():
            print("no cuda")
            continue
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        compute_throughput = compute_throughput_cuda

    if device == 'mps':
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            print("no mps")
            continue
        print('Running on MPS')
        compute_throughput = compute_throughput_mps

    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread')
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu

    for n, batch_size0, resolution in [
        ('CascadedViT_S', 2048, 224),
        ('CascadedViT_M', 2048, 224),
        ('CascadedViT_L', 2048, 224),
        ('CascadedViT_XL', 2048, 224),
    ]:

        if device == 'cpu':
            batch_size = 16
        # other devices can fit 2048
        else:
            batch_size = batch_size0 
        inputs = torch.randn(batch_size, 3, resolution,
                             resolution, device=device)
        model = eval(n)(num_classes=1000)
        utils.replace_batchnorm(model)
        model.to(device)
        model.eval()

        # report FLOP and param count during inference
        flop_input = torch.randn(1, 3, resolution, resolution, device=device)
        flop_analyser = FlopCountAnalysis(model, (flop_input, ))
        gflops = flop_analyser.total() / 1e9
        print(f"{n} FLOPs: {gflops} GFLOPs")
        n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        compute_throughput(n, model, device,
                           batch_size, resolution=resolution)
