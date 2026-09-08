import numpy as np
import matplotlib.pyplot as plt

# 原始数据（论文图8a的4个关键点）
T = [30, 31, 32, 33]
P = [108, 109.5, 110.5, 111.5]

# 二次多项式拟合，得到系数 a, b, c
coefs = np.polyfit(T, P, 2)
print(f"Coefficients (a, b, c): {coefs}")

# 生成平滑曲线的100个温度点（30~33均匀分布）
T_smooth = np.linspace(30, 33, 100)
# 用拟合公式计算对应的能耗值
P_smooth = np.polyval(coefs, T_smooth)

# 绘制图像
plt.plot(T_smooth, P_smooth, 'b-', linewidth=2, label='Fitted curve')
plt.scatter(T, P, color='red', s=50, label='Original data points')
plt.xlabel('Outlet Temperature (°C)')
plt.ylabel('Total Power (kW)')
plt.title('Fig 8a Reproduction (June)')
plt.grid(True)
plt.legend()

# 【关键】先保存，后展示
plt.savefig('Fig8a_june.png', dpi=300, bbox_inches='tight')
plt.show()