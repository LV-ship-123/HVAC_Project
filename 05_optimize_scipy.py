"""
文件名: 05_optimize_scipy.py
功能: 使用专业的 scipy 优化器，在30到33度之间寻找能耗最低的温度点。
作者: june
日期: 2026-09-16
"""
import numpy as np                     # 请数学工具箱 numpy 进来，起外号叫 np
from scipy.optimize import minimize    # 从 scipy 工具箱的 optimize 模块里，单独请出 minimize（求最小值）这个工具

# 1. 拟合系数
coefs = [-0.125, 9.025, -50.225]       # 把上一步算出的三个系数（a, b, c）装进列表里

# 2. 定义能耗函数
def power_consumption(x):              # 定义一台叫“能耗计算器”的机器，原料是 x（温度）
    return coefs[0] * x**2 + coefs[1] * x + coefs[2]  # 把温度 x 代入公式 y=ax²+bx+c，算出能耗结果丢出去

# 3. 使用 scipy 优化器，在 [30, 33] 区间内找最小值
result = minimize(power_consumption, x0=31.5, bounds=[(30.0, 33.0)], method='L-BFGS-B')  # 让工具去自动寻找最小值，结果装进 result 盒子里

# 4. 输出结果
print(f"Scipy Optimizer - Optimal Temp: {result.x[0]:.2f} °C")    # 打印：最优温度是多少度（从result盒子里取x的第一个值）
print(f"Scipy Optimizer - Minimum Power: {result.fun:.2f} kW")     # 打印：最低能耗是多少千瓦（从result盒子里取fun的值）
print(f"Optimization success: {result.success}")                  # 打印：优化是否成功（True代表成功，False代表失败）
print(f"Optimization message: {result.message}")                  # 打印：优化器给出的结束说明（不用管内容，知道它汇报了信息就行）