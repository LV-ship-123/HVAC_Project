import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# 物理模型（原始版本）
def total_power_physical(T_out, slope=5.0, pump_strength=20.0):
    P_comp = 110 + slope * (T_out - 30)
    P_pump = pump_strength / (T_out - 29.4)
    return P_comp + P_pump

# 鲁棒性测试设置
n_trials = 50
optimal_temps = []

print("🔬 开始鲁棒性测试：对物理模型参数进行随机扰动...")
for i in range(n_trials):
    # 对压缩机斜率（±20%）和水泵强度（±20%）施加随机干扰
    perturbed_slope = 5.0 * np.random.uniform(0.8, 1.2)
    perturbed_pump = 20.0 * np.random.uniform(0.8, 1.2)

    # 用遗传算法搜索最优点
    result = differential_evolution(
        lambda x: total_power_physical(x[0], slope=perturbed_slope, pump_strength=perturbed_pump),
        bounds=[(30.0, 33.0)],
        strategy='best1bin',
        maxiter=500,
        popsize=15,
        tol=0.01,
        seed=None  # 每次随机
    )
    optimal_temps.append(result.x[0])
    print(f"  第 {i+1:2d} 次: 扰动后最优温度 = {result.x[0]:.4f} °C")

# --- 统计分析 ---
temps = np.array(optimal_temps)
mean_temp = np.mean(temps)
std_temp = np.std(temps)
min_temp = np.min(temps)
max_temp = np.max(temps)

print("\n" + "="*60)
print("📊 鲁棒性测试结果 (50次随机扰动)")
print("="*60)
print(f"平均最优温度: {mean_temp:.4f} °C")
print(f"标准差:       {std_temp:.4f} °C")
print(f"最小值:       {min_temp:.4f} °C")
print(f"最大值:       {max_temp:.4f} °C")
print(f"波动范围:     [{min_temp:.4f}, {max_temp:.4f}]")
print(f"95%置信区间:  [{mean_temp - 1.96*std_temp:.4f}, {mean_temp + 1.96*std_temp:.4f}]")
print("="*60)

# --- 绘制分布直方图 ---
plt.figure(figsize=(10, 5))
plt.hist(temps, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(mean_temp, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_temp:.2f}°C')
plt.axvline(31.4, color='green', linestyle='-', linewidth=1.5, label='Original Optimum (31.40°C)')
plt.xlabel('Optimal Temperature (°C)')
plt.ylabel('Frequency')
plt.title('Robustness Test: Optimal Temperature Distribution under Parameter Perturbation (june)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('09_Robustness_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("📸 鲁棒性分布图已保存为: 09_Robustness_Distribution.png")