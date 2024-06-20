import sys

# 获取模型对象的内存大小（以字节为单位）
model_memory_usage = sys.getsizeof(model)
print("模型占用的内存大小（字节）:", model_memory_usage)
