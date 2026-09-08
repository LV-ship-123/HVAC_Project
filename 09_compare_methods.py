import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# --- 1. 物理模型（与 08 一致） ---
def total_power_physical(T_out):
    T_wb = 25.0
    P_comp = 110 + 5 * (T_out - 30)
    P_pump = 20.0 / (T_out - 29.4)
    return P_comp + P_pump

# --- 2. 用 L-BFGS-B 梯度优化器求解 ---
result_lbfgs = minimize(
    total_power_physical,
    x0=31.5,
    bounds=[(30.0, 33.0)],
    method='L-BFGS-B'
)
opt_temp_lbfgs = result_lbfgs.x[0]
opt_power_lbfgs = result_lbfgs.fun

# --- 3. 用遗传算法 (GA) 求解 ---
result_ga = differential_evolution(
    total_power_physical,
    bounds=[(30.0, 33.0)],
    strategy='best1bin',
    maxiter=500,
    popsize=20,
    tol=0.01,
    seed=42
)
opt_temp_ga = result_ga.x[0]
opt_power_ga = result_ga.fun

# --- 4. 画图对比 ---
T_range = np.linspace(30.0, 33.0, 100)
P_range = [total_power_physical(t) for t in T_range]

plt.figure(figsize=(10, 6))
# 物理模型曲线
plt.plot(T_range, P_range, 'b-', linewidth=2, label='Physical Model (Compressor + Pump)')

# L-BFGS-B 最优点（绿色菱形）
plt.scatter(opt_temp_lbfgs, opt_power_lbfgs, color='green', s=150, marker='D',
            label=f'L-BFGS-B Optimum ({opt_temp_lbfgs:.2f}°C, {opt_power_lbfgs:.2f} kW)')

# GA 最优点（红色星形）
plt.scatter(opt_temp_ga, opt_power_ga, color='red', s=200, marker='*',
            label=f'GA Optimum ({opt_temp_ga:.2f}°C, {opt_power_ga:.2f} kW)')

plt.xlabel('Cooling Tower Outlet Temperature (°C)')
plt.ylabel('Total Power Consumption (kW)')
plt.title('Optimization Method Comparison on Physical Model (june)')
plt.grid(True)
plt.legend()
plt.savefig('08_Method_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 5. 终端输出对比结果 ---
print("\n" + "="*60)
print("优化方法对比结果 (物理模型)")
print("="*60)
print(f"L-BFGS-B  : 最优温度 = {opt_temp_lbfgs:.4f} °C, 能耗 = {opt_power_lbfgs:.4f} kW")
print(f"GA        : 最优温度 = {opt_temp_ga:.4f} °C, 能耗 = {opt_power_ga:.4f} kW")
print(f"两者差值  : 温度差 = {abs(opt_temp_lbfgs - opt_temp_ga):.4f} °C, 能耗差 = {abs(opt_power_lbfgs - opt_power_ga):.4f} kW")
print("="*60)