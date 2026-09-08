import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# --- 1. 物理常数 ---
Cp_w = 4.18       # 水的比热容 (kJ/(kg·°C))
Cp_air = 1.005    # 空气的比热容 (kJ/(kg·°C))
T_wb = 25.0       # 进风湿球温度 (°C)
T_water_in = 37.0 # 进水温度 (°C)

def simulate_cooling_tower(m_water, m_air):
    """
    真正的能量平衡模型：
    空气流量直接决定换热能力。
    如果空气流量小，冷却效果差，出水温度接近进水温度。
    如果空气流量大，冷却效果好，出水温度接近湿球温度。
    """
    # 1. 计算水侧放热潜力（从37°C降到25°C）
    Q_water_max = m_water * Cp_w * (T_water_in - T_wb)
    
    # 2. 计算空气侧吸热潜力（从25°C升到37°C）
    Q_air_max = m_air * Cp_air * (T_water_in - T_wb)
    
    # 3. 实际换热量 = 水侧潜力 和 空气侧潜力 的较小值（受限）
    # 这才是物理真相：谁少，谁就决定了换热量！
    Q_actual = min(Q_water_max, Q_air_max)
    
    # 4. 如果空气流量极小，Q_actual 主要由空气侧决定
    # 计算实际出水温度：水温降低量 = Q_actual / (m_water * Cp_w)
    T_out = T_water_in - Q_actual / (m_water * Cp_w)
    
    # 物理安全限制
    if T_out < T_wb:
        T_out = T_wb
    if T_out > T_water_in:
        T_out = T_water_in
    
    # 5. 能耗计算 (真实风扇定律和水泵定律)
    # 水泵能耗：与流量成正比（克服管道阻力）
    P_pump = 0.8 * m_water  
    
    # 风机能耗：与空气流量立方成正比（真实风扇定律）
    P_fan = 0.15 * (m_air ** 3)
    
    # 6. 惩罚项：如果出水温度太高（>33°C），说明冷却不足，施加高额惩罚
    penalty = 0
    if T_out > 33.0:
        penalty = (T_out - 33.0) * 1000
    if T_out < 29.0:
        # 出水温度太低通常意味着能耗极高（空气或水流量过大），已经体现在能耗里了
        pass
    
    P_total = P_pump + P_fan + penalty
    return T_out, P_total

# --- 2. 遗传算法优化 ---
def objective(x):
    m_water, m_air = x
    T_out, P_total = simulate_cooling_tower(m_water, m_air)
    return P_total

print("🚀 开始优化真正的物理模型（能量平衡法）...")
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
print("✅ 遗传算法优化结果 (真实能量平衡模型)")
print("="*60)
print(f"最优水流量   : {best_m_water:.3f} kg/s")
print(f"最优空气流量 : {best_m_air:.3f} kg/s")
print(f"对应出水温度 : {best_T_out:.2f} °C")
print(f"最低总能耗   : {best_P_total:.2f} kW")
print(f"收敛状态     : {result.success}")
print("="*60)

# --- 3. 可视化：固定最优空气流量，看水流量对温度和能耗的影响 ---
m_water_range = np.linspace(2.0, 15.0, 50)
T_out_list = []
P_list = []

print("\n正在绘制水流量对性能的影响...")
for m_w in m_water_range:
    T_out, P = simulate_cooling_tower(m_w, best_m_air)
    T_out_list.append(T_out)
    P_list.append(P)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(m_water_range, T_out_list, 'b-', linewidth=2)
plt.scatter(best_m_water, best_T_out, color='red', s=100, marker='*', label='Optimum')
plt.xlabel('Water Flow Rate (kg/s)')
plt.ylabel('Outlet Temperature (°C)')
plt.title('Cooling Tower Performance (True Physical Model)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(m_water_range, P_list, 'g-', linewidth=2)
plt.scatter(best_m_water, best_P_total, color='red', s=100, marker='*', label='Optimum')
plt.xlabel('Water Flow Rate (kg/s)')
plt.ylabel('Total Power Consumption (kW)')
plt.title('Energy Consumption vs Water Flow')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('12_True_Physical_Optimization.png', dpi=300)
plt.show()

print("📸 真实物理仿真图像已保存为: 12_True_Physical_Optimization.png")
