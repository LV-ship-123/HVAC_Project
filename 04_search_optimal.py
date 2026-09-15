"""
文件名: 04_search_optimal.py
功能: 用最笨的“暴力穷举法”，一步步试出能耗最低的温度点。
作者: june
日期: 2026-09-15
"""
import numpy as np  # 请数学工具包 numpy 来帮忙算数，给它起个外号叫 np

# 1. 拟合系数（来自之前的结果）
# 对应公式：y = a*x² + b*x + c
coefs = [-0.125, 9.025, -50.225]  # 把上一步拟合算出来的三个系数（a, b, c）存进列表里

# 2. 定义能耗函数（把数学公式封装起来）
def power_consumption(x):  # def=定义函数，给函数起名叫“能耗计算”，括号里的 x 是原料（温度）
    return coefs[0] * x**2 + coefs[1] * x + coefs[2]  # 把温度 x 代入公式 y=ax²+bx+c，算出来的结果丢回去

# ---------- 穷举法（网格搜索）核心逻辑 ----------
# 生成搜索范围：从 30.00 到 33.00，步长 0.01
temp_range = np.arange(30.0, 33.01, 0.01)  # np.arange=生成一个等差数组，从30开始，到33.01（为了包含33），每次增加0.01

# 初始化（先假设第一个温度就是最优的）
optimal_temp = temp_range[0]  # 把搜索范围里的第一个温度值（30.0）拿出来，暂时当成“最优温度”
min_power = power_consumption(optimal_temp)  # 算一下这个假设的温度，对应的能耗是多少，先记成“最小能耗”

# 遍历每一个温度点（这就是“穷举”的意思：一个不漏地试）
for t in temp_range:  # for=循环，把 temp_range 里的每一个温度，依次赋值给变量 t
    p = power_consumption(t)  # 计算当前温度 t 下的能耗，记作 p
    if p < min_power:  # if=如果... p 比目前记录的最小能耗（min_power）还小吗？
        min_power = p  # 如果是，那就更新记录：把更小的 p 记为新的最小能耗
        optimal_temp = t  # 同时，把当前温度 t 记为新的最优温度

# 输出结果（全部用英文，符合学术规范）
print(f"Exhaustive Search - Optimal Temp: {optimal_temp:.2f} °C")    # 打印：最优温度是多少度（保留两位小数）
print(f"Exhaustive Search - Minimum Power: {min_power:.2f} kW")     # 打印：最低能耗是多少千瓦（保留两位小数）