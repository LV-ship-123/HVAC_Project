import numpy as np

# 1. 拟合系数 (来自之前的结果)
# 对应公式: y = a*x^2 + b*x + c
coefs = [-0.125, 9.025, -50.225]

# 2. 定义能耗函数 (把数学公式封装起来)
def power_consumption(x):
    return coefs[0] * x**2 + coefs[1] * x + coefs[2]

# ---------- 穷举法（网格搜索）核心逻辑 ----------

# 生成搜索范围：从 30.00 到 33.00，步长 0.01
temp_range = np.arange(30.0, 33.01, 0.01) 

# 初始化（先假设第一个温度就是最优的）
optimal_temp = temp_range[0]      # 记录最优温度
min_power = power_consumption(optimal_temp)  # 记录最小能耗

# 遍历每一个温度点（这就是“穷举”的意思：一个不漏地试）
for t in temp_range:
    p = power_consumption(t)      # 计算当前温度下的能耗
    if p < min_power:             # 如果发现了更低的能耗
        min_power = p             # 更新最低能耗记录
        optimal_temp = t          # 更新最优温度记录

# 输出结果（全部用英文，符合学术规范）
print(f"Exhaustive Search - Optimal Temp: {optimal_temp:.2f} °C")
print(f"Exhaustive Search - Minimum Power: {min_power:.2f} kW")