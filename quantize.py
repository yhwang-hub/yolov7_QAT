import sys
import os
import re

# Add the current directory to PYTHONPATH for YoloV7
sys.path.insert(0, os.path.abspath("."))
pydir = os.path.dirname(__file__)

import yaml
import collections
import warnings
import argparse
import json
from tqdm import tqdm
from pathlib import Path

# PyTorch
import torch
import torch.nn as nn

# YoloV7
import test
from models.yolo import Model
from models.common import Conv
from utils.datasets import create_dataloader
from utils.google_utils import attempt_download
from utils.general import init_seeds

from pytorch_quantization import quant_modules
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as quant_nn
from absl import logging as quant_logging

# Disable all warning
warnings.filterwarnings("ignore")

# Load YoloV7 Model
def load_yolov7_model(weight, device) -> Model:

    attempt_download(weight)
    model = torch.load(weight, map_location=device)["model"]
    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            
    model.float()
    model.eval()

    with torch.no_grad():
        model.fuse()
    return model

def create_coco_train_dataloader(cocodir, batch_size = 10):
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader = yaml.SafeLoader)
    
    loader = create_dataloader(
        f"{cocodir}/train2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=True, hyp=hyp, rect=False, cache=False, stride=32, pad=0, image_weights=False)[0]

    return loader

def create_coco_val_dataloader(cocodir, batch_size = 10, keep_images = None):
    loader = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False)[0]

    def subclass_len(self):
        if keep_images is not None:
            return keep_images
        return len(self.img_files)

    loader.dataset.__len__ = subclass_len

    return loader

def evaluate_coco(model, dataloader, using_cocotools = False, save_dir = ".", conf_thres = 0.001, iou_thres = 0.65):
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    return test.test(
        "data/coco.yaml",
        save_dir=Path(save_dir),
        dataloader=dataloader,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        is_coco=True,
        plots=False,
        half_precision=True,
        save_json=using_cocotools)[0][3]

def transfer_torch_to_quantization(nn_instance, quant_module):
    quant_instance = quant_module.__new__(quant_module)

    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        # 返回两个QuantDescriptor的实例, self.__class_是quant_instance的类(Ex: QuantConv2d)
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True # 为了加速量化
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance

def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False

    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer]
        if path in ignore_layer:
            return True
        for item in ignore_layer:
            if re.match(item, path):
                return True

    return False

# 递归函数
def torch_module_find_quant_module(module, module_dict, ignore_layer, prefix = ''):
    for name in module._modules: # 遍历module模块的子模块
        submodule = module._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_dict, ignore_layer, prefix=path)

        submodule_id = id(type(submodule))
        if submodule_id in module_dict:
            ignored = quantization_ignore_match(ignore_layer, path)
            if ignored:
                print(f"Quantization : {path} has ignored.")
                continue
            # 转换
            module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

def replace_to_quantization_model(model, ignore_layer=None):
    module_dict = {}

    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name) #获取torch模块的方法和name
        module_dict[id(module)] = entry.replace_mod

    torch_module_find_quant_module(model, module_dict, ignore_layer)

# Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR) # 保存错误日志

def collect_stats(model, dataloader, device, num_batch=200):
    # 将模型设置为评估（推理）模式。这在PyTorch中很重要，因为某些层（如Dropout和BatchNorm）在训练和评估时有不同的行为。
    model.eval()

    # 开启校准器
    # 遍历模型中的所有模块。`named_modules()`方法提供了一个迭代器，按层次结构列出模型的所有模块及其名称。
    for name, module in model.named_modules():
        # 检查当前模块是否为TensorQuantizer类型，即我们想要量化的特定类型的层。
        if isinstance(module, quant_nn.TensorQuantizer):
            # 如果此层配备了校准器。
            if module._calibrator is not None:
                # 禁用量化。这意味着层将正常（未量化）运行，使校准器能够收集必要的统计数据。
                module.disable_quant()
                # 启用校准。这使得校准器开始在此层的操作期间收集数据。
                module.enable_calib()
            else:
                # 如果没有校准器，简单地禁用量化功能，但不进行数据收集。
                module.disable()

    # 在此阶段，模型准备好接收数据，并通过处理未量化的数据来进行校准。

    # test
    # 关闭自动求导系统。这在进行推理时是有用的，因为它减少了内存使用量，加速了计算，而且我们不需要进行反向传播。
    with torch.no_grad():
        # 遍历数据加载器。数据加载器将提供批量的数据，通常用于训练或评估。
        for i, datas in tqdm(enumerate(dataloader), total=num_batch, desc="Collect stats for calibrating"):
            # 获取图像数据，转换为适当的设备（例如GPU），并将其类型转换为float。除以255是常见的归一化技术，用于将像素值缩放到0到1的范围。
            imgs = datas[0].to(device, non_blocking=True).float() / 255.0
            # 用当前批次的图像数据执行模型推理。
            model(imgs)

            # 如果我们已经处理了指定数量的批次，则停止迭代。
            if i >= num_batch:
                break

    # 关闭校准器
    # 再次遍历所有模块，就像我们之前做的那样。
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # 如果有校准器。
            if module._calibrator is not None:
                # 重新启用量化。现在，校准器已经收集了足够的统计数据，我们可以再次量化层的操作。
                module.enable_quant()
                # 禁用校准。数据收集已经完成，因此我们关闭校准器。
                module.disable_calib()
            else:
                # 如果没有校准器，我们只需重新启用量化功能。
                module.enable()

    # 在此阶段，校准过程完成，模型已经准备好以量化的状态进行更高效的运行。

def compute_amax(model, device, **kwargs):
    # 遍历模型中的所有模块，`model.named_modules()`方法提供了一个迭代器，包含模型中所有模块的名称和模块本身。
    for name, module in model.named_modules():
        # 检查当前模块是否为TensorQuantizer的实例，这是处理量化的部分。
        if isinstance(module, quant_nn.TensorQuantizer):
            # （这里的print语句已被注释掉，如果取消注释，它将打印当前处理的模块的名称。）
            # print(name)

            # 检查当前的量化模块是否具有校准器。
            if module._calibrator is not None:
                # 如果该模块的校准器是MaxCalibrator的实例（一种特定类型的校准器）...
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    # ...则调用load_calib_amax()方法，该方法计算并加载适当的'amax'值，它是量化过程中用于缩放的最大激活值。
                    module.load_calib_amax()
                else:
                    # ...如果校准器不是MaxCalibrator，我们仍然调用load_calib_amax方法，但是可以传递额外的关键字参数。
                    # 这些参数可能会影响'amax'值的计算。
                    # ['entropy', 'mse', 'percentile']   这里有三个计算方法，实际过程中要看哪一个比较准，再考虑用哪一个
                    module.load_calib_amax(**kwargs)
                # 将计算出的'amax'值（现在存储在模块的'_amax'属性中）转移到指定的设备上。
                # 这确保了与模型数据在同一设备上的'amax'值，这对于后续的计算步骤（如训练或推理）至关重要。
                module._amax = module._amax.to(device)

def calibrate_model(model, dataloader, device):
    # 收集每一层的信息
    collect_stats(model, dataloader, device, num_batch=25)
    #获取动态范围，计算amax值，scale值
    compute_amax(model, device, method = 'mse')

def export_onnx(model, save_file, device, dynamic_batch):
    input_dummy = torch.randn(1, 3, 640, 640, device=device)

    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_dummy,
            save_file,
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input' : {0: 'batch'},
                'output' : {0: 'batch'}
            } if dynamic_batch else None
        )

    quant_nn.TensorQuantizer.use_fb_fake_quant = False
    print("sucessfully export yolov7 ptq onnx!")

# 判断层是否是量化层
def have_quantizer(layer):
    for name, module in layer.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True

# 关闭量化
class disable_quantization():
    # 初始化
    def __init__(self, model) -> None:
        self.model = model

    # 应用 关闭量化
    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(disabled=True)

    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)

# 重启量化
class enable_quantization():
    def __init__(self, model) -> None:
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
                
    def __enter__(self):
        self.apply(enabled=True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)

# 日志保存
class SummaryTool():
    def __init__(self, file) -> None:
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)

def sensitive_analysis(model, dataloader, summary_file):
    summary = SummaryTool(summary_file)

    # for循环每一个层
    print("Sensitive analysis by each layer......")

    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断layer是否是量化层
        if have_quantizer(layer): # 如果是量化层
            # 使该层的量化失效，不尽兴int8的量化，使用fp16进行运算
            disable_quantization(layer).apply()

            # 计算map值
            ap = evaluate_coco(model, dataloader)

            # 保存精度值，json文件保存
            summary.append([ap, f"model.{i}"])

            # 重启该层的量化，还原
            enable_quantization(layer).apply()

            print(f"layer {i} ap: {ap}")

        #重启该层的量化
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    # 循环结束，打印前10个影响比较大的层
    summary = sorted(summary.data, key=lambda x : x[0], reverse=True)
    print("Sensitive summary: ")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: using fp16 {name}: ap = {ap:.5f}")
        summary.append([name, f"Top{n}: Using fp16 {name}, ap = {ap:.5f}"])