import numpy as np
from scipy.optimize import differential_evolution

# 1. 拟合系数（来自之前的结果）
coefs = [-0.125, 9.025, -50.225]

# 2. 定义能耗函数
def power_consumption(x):
    return coefs[0] * x**2 + coefs[1] * x + coefs[2]

# 3. 遗传算法求解（在 [30, 33] 区间内）
# differential_evolution 是一种全局优化算法，模拟生物进化过程
result = differential_evolution(
    power_consumption,           # 目标函数
    bounds=[(30.0, 33.0)],       # 搜索区间
    strategy='best1bin',         # 进化策略（默认即可）
    maxiter=1000,                # 最大迭代次数
    popsize=15,                  # 种群大小（每代有多少个体）
    tol=0.01,                    # 收敛容差
    seed=42                      # 随机种子（让结果可复现）
)

# 4. 输出结果
print(f"GA - Optimal Temp: {result.x[0]:.2f} °C")
print(f"GA - Minimum Power: {result.fun:.2f} kW")
print(f"GA - Success: {result.success}")
print(f"GA - Message: {result.message}")
print(f"GA - Number of function evaluations: {result.nfev}")