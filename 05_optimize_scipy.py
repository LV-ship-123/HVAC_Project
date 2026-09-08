import numpy as np
from scipy.optimize import minimize

# 1. 拟合系数
coefs = [-0.125, 9.025, -50.225]

# 2. 定义能耗函数
def power_consumption(x):
    return coefs[0] * x**2 + coefs[1] * x + coefs[2]

# 3. 使用 scipy 优化器，在 [30, 33] 区间内找最小值
result = minimize(power_consumption, x0=31.5, bounds=[(30.0, 33.0)], method='L-BFGS-B')

# 4. 输出结果
print(f"Scipy Optimizer - Optimal Temp: {result.x[0]:.2f} °C")
print(f"Scipy Optimizer - Minimum Power: {result.fun:.2f} kW")
print(f"Optimization success: {result.success}")
print(f"Optimization message: {result.message}")