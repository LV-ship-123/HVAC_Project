"""
文件名: 01_plot_fig8a.py
功能: 复现论文图8a，用二次多项式拟合4个数据点，画出曲线。
作者: june
日期: 2026-09-15
"""
import numpy as np          # 请 numpy 来帮忙做数学计算，给它起个小名叫 np
import matplotlib.pyplot as plt  # 请 matplotlib 来帮忙画图，给它起个小名叫 plt

# 原始数据（论文图8a的4个关键点）
T = [30, 31, 32, 33]        # 把这4个温度值装进一个叫 T 的列表里
P = [108, 109.5, 110.5, 111.5]  # 把这4个能耗值装进一个叫 P 的列表里

# 二次多项式拟合，得到系数 a, b, c
coefs = np.polyfit(T, P, 2)  # 让 np 帮我找一条 y = ax² + bx + c 的曲线，尽量穿过这4个点，算出 a、b、c
print(f"Coefficients (a, b, c): {coefs}")  # 把算出来的系数打印在终端里给我看一眼

# 生成平滑曲线的100个温度点（30~33均匀分布）
T_smooth = np.linspace(30, 33, 100)  # 从30到33度，均匀地生成100个数字，装进 T_smooth 里
# 用拟合公式计算对应的能耗值
P_smooth = np.polyval(coefs, T_smooth)  # 把这100个温度点代入刚才算出的公式，算出对应的100个能耗值

# 绘制图像
plt.plot(T_smooth, P_smooth, 'b-', linewidth=2, label='Fitted curve')  # 用蓝色实线画拟合曲线，线宽2
plt.scatter(T, P, color='red', s=50, label='Original data points')  # 用红色散点标出原始的那4个数据点
plt.xlabel('Outlet Temperature (°C)')  # 给横坐标起个名字：出水温度
plt.ylabel('Total Power (kW)')         # 给纵坐标起个名字：总能耗
plt.title('Fig 8a Reproduction (June)')  # 给整张图加个标题
plt.grid(True)                         # 给图加上网格线，方便看数值
plt.legend()                           # 把图例显示出来（用来区分蓝线是啥、红点是啥）

# 【关键】先保存，后展示
plt.savefig('Fig8a_june.png', dpi=300, bbox_inches='tight')  # 把高清图存到电脑里，名字叫 Fig8a_june.png
plt.show()                             # 最后把图画出来给我看