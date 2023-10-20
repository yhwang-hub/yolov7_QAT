'''
module.register_forward_hook是Pytorch中用于注册前向传播钩子(hook)的方法.
hook方法是在模型的前向传播过程中的不同层或模块上执行的用户自定义函数,
允许在模型运行时访问和操作中间输出.
这对于模型解释、特征提取和其他任务非常有用
'''
import torch
import torch.nn as nn

# 定义一个示例模块
class Mymodule(nn.Module):
    def __init__(self):
        super(Mymodule, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# 创建模型的实例
model = Mymodule()

# 定义前向传播hook函数
def forward_hook(module, input, output):
    print(f"Inside forward_hook for {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")

# 注册前向传播的hook
# 对model模型的fc1层进行hook函数的注册,可以通过该hook函数获取fc1层的输入和输出
hook_handle = model.fc1.register_forward_hook(forward_hook)

# 准备输入数据并进行模型的前向传播
input_data = torch.randn(2, 10)
output = model(input_data)

# 注销前向传播hook
hook_handle.remove()



'''
# QAT：PTQ_model, origin_model =======> loss ===> update: PTQ_model

# PTQ_model, origin_model 网络层/模块 匹配对
ptq_origin_layer_pairs = [
    [ptq_layer0, origin_layer0],
    [ptq_layer1, origin_layer1],
    [ptq_layer2, origin_layer2],
    ......
]

ptq_outputs = []
origin_outputs = []

def make_layer_forward_hook(module_outputs):
    def forward_hook(module, input, output):
        module_outputs.append(output)
        
    return forward_hook

remove_handle = []

# 注册每一层的hook
for ptq_m, ori_m in ptq_origin_layer_pairs:
    remove_handle.append(ptq_m.register_forward_hook(make_layer_forward_hook(ptq_outputs)))
    remove_handle.append(ori_m.register_forward_hook(make_layer_forward_hook(origin_outputs)))


# ptq模型前向
ptq_model(imgs)

# 原始模型前向
origin_model(imgs)

# 计算ptq与origin模型的loss
loss = 0
for index, (ptq_out, ori_out) in enumerate(zip(ptq_outputs, origin_outputs)):
    loss += loss_Function(ptq_out, ori_out)

# remove hook handle
for rm in remove_handle:
    rm.remove()
'''