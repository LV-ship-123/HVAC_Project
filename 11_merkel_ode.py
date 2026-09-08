import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import differential_evolution

# --- 1. 物理常数与模型参数 ---
Cp_w = 4.18       # 水的比热容 (kJ/(kg·°C))
h_a = 50.0        # 容积传热系数 (kW/(m³·°C))，简化假设
A_v = 100.0       # 单位体积的换热面积 (m²/m³)
V_tower = 2.0     # 填料体积 (m³)

T_air_in = 28.0   # 进风干球温度 (°C)
T_wb = 25.0       # 进风湿球温度 (°C) - 这是冷却极限

# --- 2. 简化的 Merkel 方程 (水温沿填料高度变化) ---
def merkel_ode(T_w, z, m_water, m_air):
    """
    简化的冷却塔微分方程
    dT_w/dz = -(h_a * A_v / (m_water * Cp_w)) * (T_w - T_wb)
    假设空气能瞬间带走热量，逼近湿球温度。
    这是最经典的“焓差驱动力”近似。
    """
    dT_dz = - (h_a * A_v / (m_water * Cp_w)) * (T_w - T_wb)
    return dT_dz

# --- 3. 仿真函数：给定进水温度和水流量，计算出水温度和能耗 ---
def simulate_cooling_tower(m_water, m_air, T_water_in=37.0):
    """
    输入：水流量 (kg/s)，空气流量 (kg/s)
    输出：出水温度 (°C)，风机能耗 (kW)，水泵能耗 (kW)
    """
    # 求解 ODE：从填料顶部（z=0）到底部（z=V_tower）
    z_range = np.linspace(0, V_tower, 100)
    T_initial = [T_water_in]
    
    try:
        # 注意：odeint 要求参数以元组形式传递
        sol = odeint(merkel_ode, T_initial, z_range, args=(m_water, m_air))
        T_out = sol[-1][0]  # 最终出水温度
    except:
        # 如果数值求解发散，返回一个很大的惩罚值
        return 100.0, 1e6, 1e6
    
    # 物理限制：出水温度不可能低于湿球温度
    if T_out < T_wb:
        T_out = T_wb
    
    # --- 能耗计算 ---
    # 1. 水泵能耗：与流量成正比（扬程固定）
    P_pump = 0.8 * m_water  # 假设每 kg/s 水耗电 0.8 kW
    
    # 2. 风机能耗：与空气流量立方成正比（真实风扇定律）
    P_fan = 0.15 * (m_air ** 3) 
    # 防止 m_air 太小导致数值不稳定，同时限制过大的风量
    if m_air < 0.1:
        P_fan = 1e6
    
    P_total = P_pump + P_fan
    return T_out, P_total

# --- 4. 遗传算法优化：寻找最优的水流量和空气流量匹配 ---
def objective(x):
    m_water, m_air = x
    # 假设进水温度固定在 37°C（典型工况）
    T_out, P_total = simulate_cooling_tower(m_water, m_air)
    
    # 加入惩罚项：如果出水温度高于 33°C，能耗指标很差，强行惩罚
    # 这里我们既要考虑能耗，又要满足出水温度在合理范围内（30~33°C）
    penalty = 0
    if T_out > 33.0:
        penalty = (T_out - 33.0) * 1000
    if T_out < 29.0:
        penalty = (29.0 - T_out) * 1000
    
    return P_total + penalty

print("🚀 开始优化物理模型（Merkel ODE 仿真）...")
print("正在搜索最优的水流量与空气流量匹配...")

# 搜索空间：水流量 2~15 kg/s，空气流量 0.5~8 kg/s
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

# --- 5. 输出结果 ---
print("\n" + "="*60)
print("✅ 遗传算法优化结果 (Merkel 动态物理模型)")
print("="*60)
print(f"最优水流量   : {best_m_water:.3f} kg/s")
print(f"最优空气流量 : {best_m_air:.3f} kg/s")
print(f"对应出水温度 : {best_T_out:.2f} °C")
print(f"最低总能耗   : {best_P_total:.2f} kW")
print(f"收敛状态     : {result.success}")
print("="*60)

# --- 6. 可视化：分析出水温度随水流量的变化关系（在最优风量下）---
m_water_range = np.linspace(2.0, 15.0, 50)
T_out_range = []
P_range = []
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
plt.title('Cooling Tower Performance vs Water Flow')
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
plt.savefig('10_Merkel_ODE_Optimization.png', dpi=300)
plt.show()

print("📸 物理仿真图像已保存为: 10_Merkel_ODE_Optimization.png")
