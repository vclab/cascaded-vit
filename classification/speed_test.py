"""
Testing the speed of different models
"""
import os
import torch
import torchvision
import time
import timm
import onnxruntime as ort
from model.build import CascadedViT_S, CascadedViT_M, CascadedViT_L, CascadedViT_XL
import torchvision
from utils import replace_batchnorm, export_to_onnx, quantize_with_quark, get_apu_info, set_environment_variable
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
    
def compute_throughput_amd(name, onnx_path, device, batch_size, resolution, inputs_np):
    sess_opts = ort.SessionOptions()
    session  = ort.InferenceSession(onnx_path, sess_opts, providers=["VitisAIExecutionProvider"])
    iname = session.get_inputs()[0].name
    # warm-up
    t0 = time.time()
    while time.time() - t0 < T0:
        session.run(None, {iname: inputs_np})
    # timed
    count = 0
    t_start = time.time()
    while time.time() - t_start < T1:
        session.run(None, {iname: inputs_np})
        count += 1
    fps = (count * batch_size) / (time.time() - t_start)
    print(f"{name} | AMD | {fps:.1f} img/s @ bs={batch_size}")

# collect available devices
available_devices = []

if torch.cuda.is_available():
    available_devices.append('cuda:0')

if (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
    available_devices.append('mps')

if "VitisAIExecutionProvider" in ort.get_available_providers() and get_apu_info() is not None:
    available_devices.append('amd')

available_devices.append('cpu')

for device in available_devices:

    for n, batch_size0, resolution in [
        ('CascadedViT_S', 2048, 224),
        ('CascadedViT_M', 2048, 224),
        ('CascadedViT_L', 2048, 224),
        ('CascadedViT_XL', 2048, 224),
    ]:
        
        if device == "amd":
            # batch_size = batch_size0

            # model_cpu = eval(n)(num_classes=1000)

            # fp32_path = n
            # int8_base = n

            # export_to_onnx(model_cpu, fp32_path)
            # quantize_with_quark(fp32_path+".onnx", int8_base)

            # inputs_np = torch.randn(batch_size, 3, resolution, resolution).numpy()

            # compute_throughput_amd(n, f"{int8_base}_int8.onnx", device,
            #                        batch_size, resolution, inputs_np)
            continue

        # ----------------------------------------------------------
        # CPU / CUDA / MPS (PyTorch)
        # ----------------------------------------------------------

        # batch size choice
        batch_size = 16 if device == "cpu" else batch_size0

        # instantiate & move model
        model = eval(n)(num_classes=1000)
        replace_batchnorm(model)
        model = model.to(device).eval()

        # FLOPs & params
        flop_input = torch.randn(1, 3, resolution, resolution, device=device)
        gflops = FlopCountAnalysis(model, flop_input).total() / 1e9
        params = sum(p.numel() for p in model.parameters())

        print(f"\n{n} | {device}")
        print(f"  FLOPs:  {gflops:.2f} GFLOPs")
        print(f"  Params: {params/1e6:.2f} M")

        # inputs for throughput
        inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)

        # choose correct throughput function
        if device == "cpu":
            compute_throughput = compute_throughput_cpu
        elif device.startswith("cuda"):
            compute_throughputn = compute_throughput_cuda
        elif device == "mps":
            compute_throughput = compute_throughput_mps

        compute_throughput(n, model, device, batch_size, resolution=resolution)

        # if device == 'cuda':
        #     bs = batch_size0

        # if device == 'mps':

        # if device == 'cpu':
        #     os.system('echo -n "nb processors "; '
        #             'cat /proc/cpuinfo | grep ^processor | wc -l; '
        #             'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        #     print('Using 1 cpu thread')
        #     torch.set_num_threads(1)
        #     compute_throughput = compute_throughput_cpu
        
        # if device == 'amd':
        #     compute_throughput = compute_throughput_amd

        

        #     if device == 'cpu':
        #         batch_size = 16
        #     # other devices can fit 2048
        #     elif device in ('cpu', 'cuda:0', 'mps'):
        #         batch_size = batch_size0 
        #         inputs = torch.randn(batch_size, 3, resolution,
        #                             resolution, device=device)
        #         model = eval(n)(num_classes=1000)
        #         replace_batchnorm(model)
        #     # amd
        #     else:
        #         model_cpu = eval(n)(num_classes=1000).cpu().eval()
        #         onnx_fp32 = n
        #         onnx_int8 = n
        #         export_to_onnx(model_cpu, onnx_fp32)
        #         quantize_with_quark(onnx_fp32+".onnx", onnx_int8)
        #     model.to(device)
        #     model.eval()

        #     # report FLOP and param count during inference
        #     flop_input = torch.randn(1, 3, resolution, resolution, device=device)
        #     flop_analyser = FlopCountAnalysis(model, (flop_input, ))
        #     gflops = flop_analyser.total() / 1e9
        #     print(f"{n} FLOPs: {gflops} GFLOPs")
        #     n_parameters = sum(p.numel()
        #                 for p in model.parameters() if p.requires_grad)
        #     print('number of params:', n_parameters)

        #     compute_throughput(n, model, device,
        #                     batch_size, resolution=resolution)
