import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# --- 1. 物理常数与模型参数 (基于 ε-NTU 简化模型) ---
Cp_w = 4.18       # 水的比热容 (kJ/(kg·°C))
Cp_air = 1.005    # 空气的比热容 (kJ/(kg·°C))
UA = 500.0        # 总传热系数 × 换热面积 (kW/°C)

T_wb = 25.0       # 进风湿球温度 (°C)，也是理论冷却极限

def simulate_cooling_tower(m_water, m_air, T_water_in=37.0):
    """
    修正后的物理仿真：
    基于能量平衡和有限的换热能力（UA有限），空气流量不再是摆设。
    """
    # 1. 水侧放出的热量（如果冷却到湿球温度）
    Q_max = m_water * Cp_w * (T_water_in - T_wb)
    
    # 2. 空气侧能带走的热量（空气从湿球温度被加热到出水温度）
    # 为了防止空气温升无限，假设空气出口温度不能超过水入口温度
    # 空气侧最大吸热能力（假设空气被加热到水温）
    Q_air_max = m_air * Cp_air * (T_water_in - T_wb)
    
    # 3. 实际换热量：受限于 UA 换热能力和两侧流体的热容
    # 简化：实际换热量 = UA * (平均温差) * 时间，但这里用简化的“效数法”
    # 计算水侧热容率 C_water = m_water * Cp_w
    # 计算空气侧热容率 C_air = m_air * Cp_air
    C_min = min(m_water * Cp_w, m_air * Cp_air)
    
    # 如果空气流量太小，C_min 会很小，换热效率会很高但总热量有限
    # 防止除零
    if C_min < 0.001:
        return T_water_in, 1e6
    
    # 效数 (Effectiveness) 近似：假设逆流换热器
    # NTU = UA / C_min
    NTU = UA / C_min
    epsilon = 1 - np.exp(-NTU)  # 简化的效数公式
    
    # 实际换热量 = 效数 * 最大可能换热量
    Q_actual = epsilon * Q_max
    
    # 4. 计算实际出水温度
    T_out = T_water_in - Q_actual / (m_water * Cp_w)
    
    # 物理安全限制：不能低于湿球温度
    if T_out < T_wb:
        T_out = T_wb
    
    # 5. 能耗计算 (与流量相关)
    # 水泵能耗：与水流量成正比（克服管道阻力）
    P_pump = 0.8 * m_water  
    # 风机能耗：与空气流量立方成正比（风扇定律）
    P_fan = 0.15 * (m_air ** 3)
    
    # 惩罚项：确保出水温度在 30~35°C 之间（合理工况）
    penalty = 0
    if T_out > 35.0:
        penalty += (T_out - 35.0) * 1000
    if T_out < 30.0:
        # 如果出水温度太低，说明水流量太大或空气流量太大，会消耗过多能量，给予惩罚
        penalty += (30.0 - T_out) * 100
    
    P_total = P_pump + P_fan + penalty
    
    return T_out, P_total

# --- 2. 遗传算法优化 ---
def objective(x):
    m_water, m_air = x
    T_out, P_total = simulate_cooling_tower(m_water, m_air)
    return P_total

print("🚀 开始优化修正后的物理模型 (基于效数法)...")
bounds = [(2.0, 15.0), (0.5, 8.0)]

result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=150,
    popsize=20,
    tol=0.05,
    seed=42
)

best_m_water, best_m_air = result.x
best_T_out, best_P_total = simulate_cooling_tower(best_m_water, best_m_air)

print("\n" + "="*60)
print("✅ 遗传算法优化结果 (修正后物理模型)")
print("="*60)
print(f"最优水流量   : {best_m_water:.3f} kg/s")
print(f"最优空气流量 : {best_m_air:.3f} kg/s")
print(f"对应出水温度 : {best_T_out:.2f} °C")
print(f"最低总能耗   : {best_P_total:.2f} kW")
print(f"收敛状态     : {result.success}")
print("="*60)

# --- 3. 可视化修正后的效果 ---
m_water_range = np.linspace(2.0, 15.0, 50)
T_out_range = []
P_range = []

# 固定最优空气流量，观察水流量变化的影响
print("\n正在绘制水流量对性能的影响...")
for m_w in m_water_range:
    T_out, P = simulate_cooling_tower(m_w, best_m_air)
    T_out_range.append(T_out)
    P_range.append(P)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(m_water_range, T_out_range, 'b-', linewidth=2)
plt.scatter(best_m_water, best_T_out, color='red', s=100, marker='*', label='Optimum')
plt.xlabel('Water Flow Rate (kg/s)')
plt.ylabel('Outlet Temperature (°C)')
plt.title('Cooling Tower Performance (Fixed Air Flow)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(m_water_range, P_range, 'g-', linewidth=2)
plt.scatter(best_m_water, best_P_total, color='red', s=100, marker='*', label='Optimum')
plt.xlabel('Water Flow Rate (kg/s)')
plt.ylabel('Total Power Consumption (kW)')
plt.title('Energy Consumption vs Water Flow')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('11_Merkel_Fixed_Optimization.png', dpi=300)
plt.show()

print("📸 修正后的物理仿真图像已保存为: 11_Merkel_Fixed_Optimization.png")