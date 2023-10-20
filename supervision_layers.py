from copy import deepcopy
import quantize
from pytorch_quantization import nn as quant_nn

ptq_model = quantize.prepare_model("./weights/yolov7.pt", "cuda:0")

quantize.replace_to_quantization_model(ptq_model, "model\.105\.m\.(.*)")
# print("ptq_model:\n{}".format(ptq_model))

# 省略了标定步骤
origin_model = deepcopy(ptq_model).eval()
quantize.disable_quantization(origin_model).apply() # 关闭量化
# print("origin_model:\n{}".format(origin_model))

supervision_list = []
for item in ptq_model.model:
    # print(item)
    supervision_list.append(id(item)) # 获取item的id值
    
keep_idx = list(range(0, len(ptq_model.model) - 1, 1)) # 除了最后一层，所有的子模块都要取出来
keep_idx.append(len(ptq_model.model) - 2) # 保证keep_idx非空

# 判断传入的模块是否需要在QAT的时候计算损失
def match_module(name, module):
    if id(module) not in supervision_list:
        return False

    idx = supervision_list.index(id(module))

    if idx in keep_idx:
        print(f"Supervision: {name} will compute loss.....")
    else:
        print(f"Supervision: {name} not compute loss.....")
    
    return idx in keep_idx # True/False

# ptq模型和原始模型的匹配对
ptq_origin_layer_pairs = []
for ((mname, ptq_module), (oriname, ori_module)) in zip(ptq_model.named_modules(), origin_model.named_modules()):
    print("mname: ", mname)
    print("type(mname): ", type(ptq_module))
    if isinstance(ptq_module, quant_nn.TensorQuantizer):
        continue

    if match_module(mname, ptq_module):
        continue

    ptq_origin_layer_pairs.append([ptq_module, ori_module])

# print(ptq_origin_layer_pairs)