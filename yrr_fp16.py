def float32_to_bfloat16(fp32):
    # 将 FP32 转换为 BF16
    # FP32 由 1 位符号、8 位指数和 23 位尾数组成
    # BF16 由 1 位符号、8 位指数和 7 位尾数组成
    
    # 将浮点数转换为二进制表示
    fp32_bits = int.from_bytes(fp32.hex().encode(), byteorder='big')
    
    # 提取符号、指数和尾数
    sign = (fp32_bits >> 31) & 0x1  # 取符号位
    exponent = (fp32_bits >> 23) & 0xFF  # 取指数位
    mantissa = fp32_bits & 0x7FFFFF  # 取尾数部分
    
    # 将尾数缩减到 7 位
    mantissa_bf16 = mantissa >> 16  # 右移 16 位以保留前 7 位尾数

    # 将 BF16 组合成字节
    bf16_bits = (sign << 15) | (exponent << 7) | mantissa_bf16
    
    # 将整形字节转换为字节数组
    return bf16_bits.to_bytes(2, byteorder='big')

# 示例
fp32_value = 1.5  # 例子中的 FP32 值
bf16_value = float32_to_bfloat16(fp32_value)

print("BF16 value in bytes:", bf16_value)
