"""
文件名: utils.py
功能: 存放公共函数（如通用拟合绘图、优化器配置等），供其他脚本调用。
作者: june
日期: 2026-09-14
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

def plot_fitting(x_data, y_data, degree=2):
    """通用拟合绘图函数：输入数据，输出拟合曲线图"""
    coefs = np.polyfit(x_data, y_data, degree)
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    y_smooth = np.polyval(coefs, x_smooth)
    plt.plot(x_smooth, y_smooth, 'b-', label='Fitted curve')
    plt.scatter(x_data, y_data, color='red', label='Original data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()
    return coefs
def my_fitting_tool(x_data, y_data, degree=2):
    """
    输入: x_data, y_data, degree
    输出: 拟合系数
    说明: 封装多项式拟合和绘图逻辑
    """
