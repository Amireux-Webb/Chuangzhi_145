import struct

def convert_fp32_to_bf16(values):
    """
    将单个浮点数或列表从 FP32 转换为 BF16 格式。
    
    参数:
        values (float or list): 单个浮点数或浮点数列表
    
    返回:
        list: 每个值对应 (BF16 截断二进制, 恢复为 FP32 的值)
    """
    def fp32_to_bf16(fp32_value):
        # 将 FP32 的值转换为二进制表示
        fp32_binary = struct.unpack('>I', struct.pack('>f', fp32_value))[0]
        # 提取高16位，低16位被截断
        bf16_binary = (fp32_binary >> 16) & 0xFFFF
        # 如果需要重新还原为 FP32 格式，则填补低16位为零
        bf16_to_fp32_binary = (bf16_binary << 16)
        bf16_to_fp32_value = struct.unpack('>f', struct.pack('>I', bf16_to_fp32_binary))[0]
        return bf16_binary, bf16_to_fp32_value

    # 判断输入是单个值还是列表
    if isinstance(values, (float, int)):
        return fp32_to_bf16(values)
    elif isinstance(values, list):
        return [fp32_to_bf16(value) for value in values]
    else:
        raise ValueError("输入必须是浮点数或浮点数列表")

# 示例测试
inputs = [1.5, -2.0, 0.1, 3.141592]
results = convert_fp32_to_bf16(inputs)

for original, (bf16_binary, bf16_value) in zip(inputs, results):
    print(f"Original FP32: {original}")
    print(f"BF16 Binary: {bf16_binary:016b}")
    print(f"Restored FP32: {bf16_value}")
    print("-" * 30)
