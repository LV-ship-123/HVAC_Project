import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# --- 1. 物理常数 ---
Cp_w = 4.18
Cp_air = 1.005
T_wb = 25.0
T_water_in = 37.0

def simulate_cooling_tower(m_water, m_air):
    """
    真实能量平衡 + 工程可行域惩罚（让低流量变得昂贵）
    """
    # 1. 计算换热
    Q_water_max = m_water * Cp_w * (T_water_in - T_wb)
    Q_air_max = m_air * Cp_air * (T_water_in - T_wb)
    Q_actual = min(Q_water_max, Q_air_max)
    T_out = T_water_in - Q_actual / (m_water * Cp_w)
    
    # 物理安全限制
    if T_out < T_wb:
        T_out = T_wb
    if T_out > T_water_in:
        T_out = T_water_in
    
    # 2. 能耗计算
    P_pump = 0.8 * m_water  
    P_fan = 0.15 * (m_air ** 3)
    
    # 3. 工程惩罚（新加入的“驱赶”机制）
    # 冷却塔在低流量时性能急剧恶化，用分段函数模拟
    penalty = 0
    
    # 惩罚1：水流量太低（< 4 kg/s），导致冷却效果不可靠
    if m_water < 4.0:
        penalty += 200.0 * (4.0 - m_water) ** 2  # 二次惩罚，越界越痛
    
    # 惩罚2：出水温度过高（> 33.5°C），系统效率下降
    if T_out > 33.5:
        penalty += 100.0 * (T_out - 33.5) ** 2
    
    # 惩罚3：出水温度过低（< 28°C），意味着水量太大，浪费能量
    if T_out < 28.0:
        penalty += 50.0 * (28.0 - T_out) ** 2
    
    P_total = P_pump + P_fan + penalty
    return T_out, P_total

# --- 2. 遗传算法优化 ---
def objective(x):
    m_water, m_air = x
    _, P_total = simulate_cooling_tower(m_water, m_air)
    return P_total

print("🚀 开始优化带工程惩罚的物理模型...")
bounds = [(2.0, 15.0), (0.5, 8.0)]

result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=200,
    popsize=20,
    tol=0.05,
    seed=42
)

best_m_water, best_m_air = result.x
best_T_out, best_P_total = simulate_cooling_tower(best_m_water, best_m_air)

print("\n" + "="*60)
print("✅ 遗传算法优化结果 (带工程惩罚)")
print("="*60)
print(f"最优水流量   : {best_m_water:.3f} kg/s")
print(f"最优空气流量 : {best_m_air:.3f} kg/s")
print(f"对应出水温度 : {best_T_out:.2f} °C")
print(f"最低总能耗   : {best_P_total:.2f} kW")
print(f"收敛状态     : {result.success}")
print("="*60)

# --- 3. 可视化 ---
m_water_range = np.linspace(2.0, 15.0, 100)
T_out_list = []
P_list = []

print("正在绘制水流量对性能的影响...")
for m_w in m_water_range:
    T_out, P = simulate_cooling_tower(m_w, best_m_air)
    T_out_list.append(T_out)
    P_list.append(P)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(m_water_range, T_out_list, 'b-', linewidth=2)
plt.scatter(best_m_water, best_T_out, color='red', s=100, marker='*', label='Optimum')
plt.axvline(4.0, color='gray', linestyle='--', alpha=0.7, label='Min reliable flow (4 kg/s)')
plt.xlabel('Water Flow Rate (kg/s)')
plt.ylabel('Outlet Temperature (°C)')
plt.title('Cooling Tower Performance (with Penalty)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(m_water_range, P_list, 'g-', linewidth=2)
plt.scatter(best_m_water, best_P_total, color='red', s=100, marker='*', label='Optimum')
plt.axvline(4.0, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Water Flow Rate (kg/s)')
plt.ylabel('Total Power Consumption (kW)')
plt.title('Energy Consumption vs Water Flow (with Penalty)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('13_Merkel_With_Penalty.png', dpi=300)
plt.show()

print("📸 带惩罚的物理仿真图像已保存为: 13_Merkel_With_Penalty.png")