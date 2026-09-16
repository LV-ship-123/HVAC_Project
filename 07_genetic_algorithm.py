"""
文件名: 07_genetic_algorithm.py
功能: 用遗传算法（GA）寻找能耗最低的温度点，验证与梯度优化器结果是否一致。
作者: june
日期: 2026-09-16
"""
import numpy as np                              # 请数学工具箱 numpy 进来，起外号叫 np
from scipy.optimize import differential_evolution # 从 scipy 优化工具箱里，请出 differential_evolution（差分进化，一种高级遗传算法）

# 1. 拟合系数（来自之前的结果）
coefs = [-0.125, 9.025, -50.225]                # 把上一步算出的三个系数（a, b, c）装进列表里

# 2. 定义能耗函数
def power_consumption(x):                       # 定义一台叫“能耗计算器”的机器，原料是 x（温度）
    return coefs[0] * x**2 + coefs[1] * x + coefs[2] # 把温度 x 代入公式 y=ax²+bx+c，算出能耗结果丢出去

# 3. 遗传算法求解（在 [30, 33] 区间内）
# differential_evolution 是一种全局优化算法，模拟生物进化过程
result = differential_evolution(                # 让 GA 工具去自动寻找最小值，结果装进 result 盒子里
    power_consumption,                          # 目标函数：告诉它算哪个函数的极值
    bounds=[(30.0, 33.0)],                      # 搜索区间：只能在30到33度之间找
    strategy='best1bin',                        # 进化策略：一种变异策略，默认即可
    maxiter=1000,                               # 最大迭代次数：最多繁衍1000代
    popsize=15,                                 # 种群大小：每一代有多少个测试点（个体）
    tol=0.01,                                   # 收敛容差：如果结果变化小于0.01，就提前结束
    seed=42                                     # 随机种子：让随机结果可复现（保证每次跑结果一样）
)

# 4. 输出结果
print(f"GA - Optimal Temp: {result.x[0]:.2f} °C")    # 打印：最优温度（从 result 盒子的 x 属性取第一个值）
print(f"GA - Minimum Power: {result.fun:.2f} kW")    # 打印：最低能耗（result 盒子的 fun 属性）
print(f"GA - Success: {result.success}")              # 打印：优化是否成功（True 或 False）
print(f"GA - Message: {result.message}")              # 打印：优化器结束时的说明信息
print(f"GA - Number of function evaluations: {result.nfev}") # 打印：算法总共算了多少次函数值（评估次数）