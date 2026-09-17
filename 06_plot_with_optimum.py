"""
文件名: 06_plot_with_optimum.py
功能: 使用 scipy 优化器寻找最低能耗点，并在拟合曲线上用绿色星星标出最优点。
作者: june
日期: 2026-09-15
"""
import numpy as np          # 请数学工具包来帮忙算数，起外号叫 np
import matplotlib.pyplot as plt  # 请画图工具包来帮忙画图，起外号叫 plt
from scipy.optimize import minimize  # 从科学计算工具包里，只拿 "minimize"（寻找最小值）这个工具

# 1. 拟合系数（直接从之前的结果拿过来）
coefs = [-0.125, 9.025, -50.225]  # 把二次曲线公式的系数 a, b, c 装进 coefs 盒子里

# 2. 定义能耗函数
def power_consumption(x):  # def 就是 define（定义），定义了一个叫 power_consumption 的函数，需要传入 x
    return coefs[0] * x**2 + coefs[1] * x + coefs[2]  # 返回二次多项式计算的结果：y = ax² + bx + c

# 3. 原始数据（论文里的4个点）
T_data = [30, 31, 32, 33]          # 横坐标：原始的4个温度值
P_data = [108, 109.5, 110.5, 111.5]  # 纵坐标：原始的4个能耗值

# 4. 使用 scipy 求解最优点（在 30~33°C 范围内）
# 调用 minimize 工具，给它传目标函数，初始值 x0=31.5，范围 bounds=[(30,33)]，方法选择 L-BFGS-B
result = minimize(power_consumption, x0=31.5, bounds=[(30.0, 33.0)], method='L-BFGS-B')
opt_temp = result.x[0]   # 从结果 (result) 里的 x 数组中，取出第1个数（即最优温度）
opt_power = result.fun   # 从结果里，取出函数的最小值（即最低能耗）

# 5. 生成拟合曲线（用于画图）
T_smooth = np.linspace(30, 33, 100)  # 在 30 到 33 度之间，均匀生成 100 个点，用来画平滑曲线
P_smooth = power_consumption(T_smooth)  # 把这 100 个点代入能耗函数，算出对应的 100 个能耗值

# 6. 绘制图像（核心：用绿色大星星标出最优点）
plt.plot(T_smooth, P_smooth, 'b-', linewidth=2, label='Fitted curve')  # 画蓝色的拟合曲线，线宽为 2，标签为"拟合曲线"
plt.scatter(T_data, P_data, color='red', s=50, label='Original data')  # 画红色原始散点图，点大小为 50，标签为"原始数据"
# 这里是重点：用绿色大星星标出最优解，并在旁边加注释
# 在 (最优温度, 最低能耗) 的位置，画一个绿色的星星，点大小为 200，形状是 '*'（星号）
plt.scatter(opt_temp, opt_power, color='green', s=200, marker='*', 
            # f"..." 是格式化字符串，把最优温度和最低能耗塞进标签里，保留2位小数
            label=f'Optimal Point ({opt_temp:.2f}°C, {opt_power:.2f} kW)')
plt.xlabel('Outlet Temperature (°C)')  # 给横坐标起名："出口温度 (摄氏度)"
plt.ylabel('Total Power (kW)')          # 给纵坐标起名："总能耗 (千瓦)"
plt.title('Fig 8a with Optimal Point (june)')  # 给整张图起标题
plt.grid(True)                          # 打开网格线
plt.legend()                            # 显示图例框（把曲线、散点、绿星星的名字汇总显示）
plt.savefig('06_fig8a_with_optimum.png', dpi=300, bbox_inches='tight')  # 保存成高清图片文件
plt.show()                              # 把图画出来给你看