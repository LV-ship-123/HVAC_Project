import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. 拟合系数
coefs = [-0.125, 9.025, -50.225]

# 2. 定义能耗函数
def power_consumption(x):
    return coefs[0] * x**2 + coefs[1] * x + coefs[2]

# 3. 原始数据（论文里的4个点）
T_data = [30, 31, 32, 33]
P_data = [108, 109.5, 110.5, 111.5]

# 4. 使用 scipy 求解最优点（在 30~33°C 范围内）
result = minimize(power_consumption, x0=31.5, bounds=[(30.0, 33.0)], method='L-BFGS-B')
opt_temp = result.x[0]
opt_power = result.fun

# 5. 生成拟合曲线（用于画图）
T_smooth = np.linspace(30, 33, 100)
P_smooth = power_consumption(T_smooth)

# 6. 绘制图像（核心：用绿色大星星标出最优点）
plt.plot(T_smooth, P_smooth, 'b-', linewidth=2, label='Fitted curve')
plt.scatter(T_data, P_data, color='red', s=50, label='Original data')
# 这里是重点：用绿色大星星标出最优解，并在旁边加注释
plt.scatter(opt_temp, opt_power, color='green', s=200, marker='*', 
            label=f'Optimal Point ({opt_temp:.2f}°C, {opt_power:.2f} kW)')
plt.xlabel('Outlet Temperature (°C)')
plt.ylabel('Total Power (kW)')
plt.title('Fig 8a with Optimal Point (june)')
plt.grid(True)
plt.legend()
plt.savefig('Fig8a_with_optimum.png', dpi=300, bbox_inches='tight')
plt.show()