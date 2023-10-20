import torch
import onnx
import os
import quantize

# 在onnx模型中查找 特定输入节点的所有节点，以列表的形式返回
def find_all_with_input_node(model, name):
    all = [] # 存储查找到的所有节点
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            all.append(node)
    return all

# 在onnx模型中查找指定输入节点的节点，找到了立刻返回
def find_with_input_node(model, name):
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            return node

# 在onnx模型中查找给定的QuantizeLinear节点相关联的Conv
def find_quantizelinear_conv(model, qnode):
    dq = find_with_input_node(model, qnode.output[0]) # 找到与q节点相关联的反量化节点

    conv = find_with_input_node(model, dq.output[0]) 

    return conv

# 在onnx模型中查找特定的输出名称的节点
def find_with_output_node(model, name):
    for node in model.graph.node:
        if len(node.output) > 0 and name in node.output:
            return node

# 在onnx模型中查找指定量化节点的相关卷积模块的名称
def find_quantize_conv_name(model, weight_qname):
    dq = find_with_output_node(model, weight_qname)
    
    q = find_with_output_node(model, dq.input[0])
    
    return ".".join(q.input[0].split(".")[:-1])
    # model.63.conv.weight ===> model.63.conv

def find_quantizer_pairs(onnx_file):       
    model = onnx.load(onnx_file)

    match_pairs = []
    for node in model.graph.node:
        if node.op_type == "Concat":
            # 找到那些将node节点的输出 node.output[0]作为其输入的所有节点
            all_nodes = find_all_with_input_node(model, node.output[0]) # concat的输出只有一个
            print("all_nodes:\n", all_nodes)

            major = None
            for qnode in all_nodes:
                if qnode.op_type != "QuantizeLinear":
                    continue

                conv = find_quantizelinear_conv(model, qnode)

                # 根据conv节点，找到torch所对应的conv模块的name
                # conv.input[1]对应权重量化，conv.input[0]对应input量化
                # conv_name = find_quantize_conv_name(model, conv.input[1])

                if major in None:
                    # major = conv_name
                    major = find_quantize_conv_name(model, conv.input[1])
                else:
                    # match_pairs.append([major, subconv_name])
                    match_pairs.append([major, find_quantize_conv_name(model, conv.input[1])])

                # 接下来查找输入的scale，节点
                for subnode in model.graph.node:
                    if len(subnode.input) > 0 and subnode.op_type == "QuantizeLinear" and subnode.input[0] in node.input:
                        subconv = find_quantizelinear_conv(model, subnode)
                        subconv_name = find_quantize_conv_name(model, subconv.input[1])

                        # 保存匹配关系
                        # match_pairs.append([conv_name, subconv_name])
                        match_pairs.append([major, subconv_name])

        elif node.op_type == "MaxPool":
            qnode = find_with_input_node(model, node.output[0])
            if not (qnode and qnode.op_type == "QuantizeLinear"):
                continue

            major = find_quantizelinear_conv(model, qnode)
            major = find_quantize_conv_name(model, major.input[1])
            same_input_nodes = find_all_with_input_node(model, node.input[0])

            for same_input_node in same_input_nodes:
                if same_input_node.op_type == "QuantizeLinear":
                    subconv = find_quantizelinear_conv(model, same_input_node)
                    match_pairs.append([major, find_quantize_conv_name(model, subconv.input[1])])

    return match_pairs

# 用于获取模块model中给定路径path的属性值
def get_attr_with_path(module, path):
    # 定义内置函数
    def sub_attr(module, names):
        name = names[0]
        
        value = getattr(module, name)

        if len(names) == 1:
            return value

        return sub_attr(value, names[1:])
    return sub_attr(module, path.split("."))

# 使用match_pairs，在model的基础上，进行scale值的替换
def apply_custom_rules_to_quantizer(qdq_model, device="cpu"):
    quantize.export_onnx(qdq_model, "custom_rules_temp.onnx", device)
    
    match_pairs = find_quantizer_pairs("custom_rules_temp.onnx")
    
    for major, sub in match_pairs:
        print(f"Rules: {sub} match to {major}")

        # 获取major的输入量化器，将其替换到sub子模块的输入量化器
        get_attr_with_path(qdq_model, sub)._input_quantizer = get_attr_with_path(qdq_model, major)._input_quantizer

        # 上面一行代码可以解释为于下面几行代码
        '''
        # 获取主模块和子模块的输入量化器
        input_quantizer_major = get_attr_with_path(qdq_model, major)._input_quantizer
        input_quantizer_sub = get_attr_with_path(qdq_model, sub)._input_quantizer
        # 将子模块的输入量化器设置为主模块的输入量化器
        input_quantizer_sub = input_quantizer_major
        '''

    # 移除temp.onnx
    os.remove("custom_rules_temp.onnx")

