import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# =============================================
# 1. 湿空气热力学性质（核心物理基础）
# =============================================
def h_s(T):
    """
    计算饱和湿空气的焓值 (kJ/kg 干空气)
    基于 Clausius-Clapeyron 近似和理想气体混合
    """
    # 水蒸气饱和压力 (kPa)
    p_ws = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    # 含湿量 (kg 水蒸气 / kg 干空气)
    w = 0.622 * p_ws / (101.325 - p_ws)
    # 焓值公式：干空气焓 + 水蒸气焓
    h = 1.006 * T + w * (2501 + 1.86 * T)
    return h

# =============================================
# 2. Merkel 方程（耦合 ODE 系统）
# =============================================
def cooling_tower_ode(z, y, m_w, m_a, K):
    """
    基于 Merkel 焓差理论的耦合微分方程
    y[0] = T_w (水温)
    y[1] = h_a (空气焓值)
    """
    T_w, h_a = y
    Cp_w = 4.18  # 水的比热容 (kJ/kg·K)

    # 计算当前温度下的饱和空气焓值
    h_sat = h_s(T_w)

    # 焓差驱动力
    driving_force = h_sat - h_a

    # 如果驱动力为负（物理上不应发生），强行截断防止震荡
    if driving_force < 0:
        driving_force = 0

    # 1. 水温变化率 (水侧放热)
    dT_w_dz = -K * driving_force / (m_w * Cp_w)

    # 2. 空气焓值变化率 (空气侧吸热，基于能量守恒)
    # m_w * Cp_w * dT_w + m_a * dh_a = 0
    dh_a_dz = -(m_w * Cp_w / m_a) * dT_w_dz

    return [dT_w_dz, dh_a_dz]

# =============================================
# 3. 仿真函数
# =============================================
def simulate_cooling_tower(m_w, m_a, T_in=37.0, T_wb=25.0, H=1.0, K=0.8):
    """
    输入：水流量 (kg/s)，空气流量 (kg/s)
    输出：出水温度 (°C)，总能耗 (kW)
    """
    # 初始条件：进水温度，进风焓值（在湿球温度下近似饱和）
    h_air_in = h_s(T_wb)
    y0 = [T_in, h_air_in]

    # 求解 ODE（z 从 0 到 H）
    sol = solve_ivp(
        cooling_tower_ode,
        [0, H],
        y0,
        args=(m_w, m_a, K),
        method='RK45',
        dense_output=False,
        rtol=1e-4,
        atol=1e-6
    )

    if not sol.success:
        # 求解失败，返回极差的结果作为惩罚
        return 100.0, 1e6

    # 提取最终结果
    T_out = sol.y[0, -1]
    h_air_out = sol.y[1, -1]

    # 物理安全：出水温度不能低于湿球温度
    if T_out < T_wb:
        T_out = T_wb

    # ----- 能耗计算 -----
    # 水泵功耗：与流量成正比 (简单假设)
    P_pump = 0.8 * m_w
    # 风机功耗：与空气流量的立方成正比 (实际风扇定律)
    P_fan = 0.12 * (m_a ** 3)

    # ----- 工程约束惩罚 -----
    penalty = 0
    # 1. 出水温度必须 <= 34°C（否则冷却效果太差）
    if T_out > 34.0:
        penalty += 200.0 * (T_out - 34.0) ** 2
    # 2. 水流量不能低于 3 kg/s（否则布水不均匀）
    if m_w < 3.0:
        penalty += 100.0 * (3.0 - m_w) ** 2
    # 3. 空气流量不能太低（否则风机失去效率）
    if m_a < 1.0:
        penalty += 50.0 * (1.0 - m_a) ** 2

    P_total = P_pump + P_fan + penalty
    return T_out, P_total

# =============================================
# 4. 遗传算法优化
# =============================================
def objective(x):
    m_w, m_a = x
    _, P_total = simulate_cooling_tower(m_w, m_a)
    return P_total

print("🚀 启动基于 Merkel 焓差模型的耦合 ODE 优化...")
bounds = [(2.0, 12.0), (0.5, 6.0)]

result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=150,
    popsize=15,
    tol=0.01,
    seed=42
)

best_m_w, best_m_a = result.x
best_T_out, best_P_total = simulate_cooling_tower(best_m_w, best_m_a)

print("\n" + "="*60)
print("✅ 遗传算法优化结果 (Merkel 焓差耦合模型)")
print("="*60)
print(f"最优水流量   : {best_m_w:.3f} kg/s")
print(f"最优空气流量 : {best_m_a:.3f} kg/s")
print(f"对应出水温度 : {best_T_out:.2f} °C")
print(f"最低总能耗   : {best_P_total:.2f} kW")
print(f"收敛状态     : {result.success}")
print("="*60)

# =============================================
# 5. 可视化
# =============================================
m_w_range = np.linspace(2.0, 12.0, 50)
T_out_list = []
P_list = []

print("\n正在绘制水流量对性能的影响...")
for m_w in m_w_range:
    T_out, P = simulate_cooling_tower(m_w, best_m_a)
    T_out_list.append(T_out)
    P_list.append(P)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(m_w_range, T_out_list, 'b-', linewidth=2)
plt.scatter(best_m_w, best_T_out, color='red', s=100, marker='*', label='Optimum')
plt.axhline(34.0, color='gray', linestyle='--', alpha=0.7, label='Max allowed T_out (34°C)')
plt.xlabel('Water Flow Rate (kg/s)')
plt.ylabel('Outlet Temperature (°C)')
plt.title('Cooling Tower Performance (Merkel ODE Coupled)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(m_w_range, P_list, 'g-', linewidth=2)
plt.scatter(best_m_w, best_P_total, color='red', s=100, marker='*', label='Optimum')
plt.xlabel('Water Flow Rate (kg/s)')
plt.ylabel('Total Power Consumption (kW)')
plt.title('Energy Consumption vs Water Flow (Merkel ODE)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('14_Merkel_ODE_Coupled.png', dpi=300)
plt.show()

print("📸 耦合 ODE 物理仿真图像已保存为: 14_Merkel_ODE_Coupled.png")