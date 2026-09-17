"""
文件名: 15_merkel_ode_coupled.py
功能: 基于 Merkel 焓差理论，建立耦合的微分方程（ODE）模型，真实模拟水温和空气焓值的同步变化。
作者: june
日期: 2026-09-18
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from scipy.integrate import solve_ivp              # 从 scipy 积分工具箱里，请出 solve_ivp（微分方程求解器）
from scipy.optimize import differential_evolution  # 从 scipy 优化工具箱里，请出 differential_evolution（遗传算法）

# =============================================
# 1. 湿空气热力学性质（核心物理基础）
# =============================================
def h_s(T):                                        # 定义机器“h_s”，原料是温度 T
    """
    计算饱和湿空气的焓值 (kJ/kg 干空气)
    基于 Clausius-Clapeyron 近似和理想气体混合
    """
    # 水蒸气饱和压力 (kPa)
    p_ws = 0.6108 * np.exp(17.27 * T / (T + 237.3)) # 计算饱和水蒸气压力，这是一个经验公式
    # 含湿量 (kg 水蒸气 / kg 干空气)
    w = 0.622 * p_ws / (101.325 - p_ws)            # 计算空气的含湿量（空气中含有多少水蒸气）
    # 焓值公式：干空气焓 + 水蒸气焓
    h = 1.006 * T + w * (2501 + 1.86 * T)          # 计算总焓值（显热 + 潜热）
    return h                                       # 把计算出的焓值丢出去

# =============================================
# 2. Merkel 方程（耦合 ODE 系统）
# =============================================
def cooling_tower_ode(z, y, m_w, m_a, K):          # 定义微分方程机器，原料是高度z、状态y、水流量、空气流量、传热系数K
    """
    基于 Merkel 焓差理论的耦合微分方程
    y[0] = T_w (水温)
    y[1] = h_a (空气焓值)
    """
    T_w, h_a = y                                   # 解包：把状态数组y拆成水温和空气焓值两个变量
    Cp_w = 4.18                                    # 水的比热容 (kJ/kg·K)

    # 计算当前温度下的饱和空气焓值
    h_sat = h_s(T_w)                               # 调用上面的h_s机器，算出当前水温对应的饱和空气焓

    # 焓差驱动力
    driving_force = h_sat - h_a                    # 焓差驱动力 = 饱和空气焓 - 实际空气焓（这是换热的根本驱动力！）

    # 如果驱动力为负（物理上不应发生），强行截断防止震荡
    if driving_force < 0:                          # 如果焓差小于0（空气中水分已经饱和，无法吸收更多水分）
        driving_force = 0                          # 强行设为0，防止电脑算出不合常理的负数

    # 1. 水温变化率 (水侧放热)
    dT_w_dz = -K * driving_force / (m_w * Cp_w)    # 水温随高度的变化率：负号代表降温，受驱动力和水流量影响

    # 2. 空气焓值变化率 (空气侧吸热，基于能量守恒)
    # m_w * Cp_w * dT_w + m_a * dh_a = 0
    dh_a_dz = -(m_w * Cp_w / m_a) * dT_w_dz        # 空气焓值随高度的变化率：由能量守恒推导，水放出的热量等于空气吸收的热量

    return [dT_w_dz, dh_a_dz]                      # 把两个变化率打包成列表返回

# =============================================
# 3. 仿真函数
# =============================================
def simulate_cooling_tower(m_w, m_a, T_in=37.0, T_wb=25.0, H=1.0, K=0.8): # 定义仿真机器，输入多组参数，默认进水37度，湿球25度
    """
    输入：水流量 (kg/s)，空气流量 (kg/s)
    输出：出水温度 (°C)，总能耗 (kW)
    """
    # 初始条件：进水温度，进风焓值（在湿球温度下近似饱和）
    h_air_in = h_s(T_wb)                           # 进风口空气焓值（在湿球温度下计算）
    y0 = [T_in, h_air_in]                          # 把进水温度和进风焓值打包成初始状态数组

    # 求解 ODE（z 从 0 到 H）
    sol = solve_ivp(                               # 调用 solve_ivp 求解微分方程
        cooling_tower_ode,                         # 方程本身
        [0, H],                                    # 求解范围：从高度0到H
        y0,                                        # 初始状态
        args=(m_w, m_a, K),                        # 额外参数（水流量、空气流量、K）
        method='RK45',                             # 使用 RK45 数值解法（Runge-Kutta 4(5)阶）
        dense_output=False,                        # 不需要密集输出
        rtol=1e-4,                                 # 相对误差容忍度（越小越精准，但越慢）
        atol=1e-6                                  # 绝对误差容忍度
    )

    if not sol.success:                            # 如果求解失败
        # 求解失败，返回极差的结果作为惩罚
        return 100.0, 1e6                          # 返回100度和100万惩罚

    # 提取最终结果
    T_out = sol.y[0, -1]                           # 提取最后一个温度值（出水温度）
    h_air_out = sol.y[1, -1]                       # 提取最后一个空气焓值（虽然在能耗里暂时没用上，但算出它利于检查物理过程）

    # 物理安全：出水温度不能低于湿球温度
    if T_out < T_wb:                               # 如果出水温度低于湿球温度
        T_out = T_wb                               # 强行设为湿球温度

    # ----- 能耗计算 -----
    # 水泵功耗：与流量成正比 (简单假设)
    P_pump = 0.8 * m_w                             # 水泵能耗 = 0.8 × 水流量
    # 风机功耗：与空气流量的立方成正比 (实际风扇定律)
    P_fan = 0.12 * (m_a ** 3)                      # 风机能耗 = 0.12 × 空气流量三次方

    # ----- 工程约束惩罚 -----
    penalty = 0                                    # 初始化惩罚值为0
    # 1. 出水温度必须 <= 34°C（否则冷却效果太差）
    if T_out > 34.0:                               # 如果出水温度大于34度
        penalty += 200.0 * (T_out - 34.0) ** 2     # 超出部分平方后乘以200作为惩罚
    # 2. 水流量不能低于 3 kg/s（否则布水不均匀）
    if m_w < 3.0:                                  # 如果水流量低于3
        penalty += 100.0 * (3.0 - m_w) ** 2        # 每低1，罚100的平方
    # 3. 空气流量不能太低（否则风机失去效率）
    if m_a < 1.0:                                  # 如果空气流量低于1
        penalty += 50.0 * (1.0 - m_a) ** 2         # 罚！

    P_total = P_pump + P_fan + penalty             # 总能耗 = 泵 + 风机 + 惩罚
    return T_out, P_total                          # 返回出水温度和总能耗

# =============================================
# 4. 遗传算法优化
# =============================================
def objective(x):                                  # 定义目标函数
    m_w, m_a = x                                   # 解包数组x
    _, P_total = simulate_cooling_tower(m_w, m_a)  # 调用仿真机器，忽略温度，只拿能耗
    return P_total                                 # 返回总能耗

print("🚀 启动基于 Merkel 焓差模型的耦合 ODE 优化...") # 打印提示
bounds = [(2.0, 12.0), (0.5, 6.0)]                 # 搜索范围：水流量2~12，空气流量0.5~6

result = differential_evolution(                   # 调用遗传算法
    objective,                                     # 目标函数
    bounds,                                        # 搜索范围
    strategy='best1bin',                           # 进化策略
    maxiter=150,                                   # 最大迭代次数
    popsize=15,                                    # 种群大小
    tol=0.01,                                      # 收敛容差
    seed=42                                        # 随机种子
)

best_m_w, best_m_a = result.x                      # 提取最优参数
best_T_out, best_P_total = simulate_cooling_tower(best_m_w, best_m_a) # 用最优参数再算一次

print("\n" + "="*60)                               # 打印换行和60个等号
print("✅ 遗传算法优化结果 (Merkel 焓差耦合模型)")       # 打印标题
print("="*60)                                      # 打印等号
print(f"最优水流量   : {best_m_w:.3f} kg/s")         # 打印最优水流量
print(f"最优空气流量 : {best_m_a:.3f} kg/s")         # 打印最优空气流量
print(f"对应出水温度 : {best_T_out:.2f} °C")         # 打印出水温度
print(f"最低总能耗   : {best_P_total:.2f} kW")       # 打印最低能耗
print(f"收敛状态     : {result.success}")            # 打印是否成功
print("="*60)                                      # 打印等号

# =============================================
# 5. 可视化
# =============================================
m_w_range = np.linspace(2.0, 12.0, 50)             # 生成50个水流量点
T_out_list = []                                    # 空列表装温度
P_list = []                                        # 空列表装能耗

print("\n正在绘制水流量对性能的影响...")                # 打印提示
for m_w in m_w_range:                              # 遍历水流量点
    T_out, P = simulate_cooling_tower(m_w, best_m_a) # 用固定最优风量算结果
    T_out_list.append(T_out)                       # 追加温度
    P_list.append(P)                               # 追加能耗

plt.figure(figsize=(12, 5))                        # 创建画布

plt.subplot(1, 2, 1)                               # 第1个子图
plt.plot(m_w_range, T_out_list, 'b-', linewidth=2) # 画蓝色曲线
plt.scatter(best_m_w, best_T_out, color='red', s=100, marker='*', label='Optimum') # 画红色星星
plt.axhline(34.0, color='gray', linestyle='--', alpha=0.7, label='Max allowed T_out (34°C)') # 画灰色水平虚线，在y=34的位置
plt.xlabel('Water Flow Rate (kg/s)')               # X轴标签
plt.ylabel('Outlet Temperature (°C)')              # Y轴标签
plt.title('Cooling Tower Performance (Merkel ODE Coupled)') # 子图标题
plt.grid(True)                                     # 网格
plt.legend()                                       # 图例

plt.subplot(1, 2, 2)                               # 第2个子图
plt.plot(m_w_range, P_list, 'g-', linewidth=2)     # 画绿色曲线
plt.scatter(best_m_w, best_P_total, color='red', s=100, marker='*', label='Optimum') # 画红色星星
plt.xlabel('Water Flow Rate (kg/s)')               # X轴标签
plt.ylabel('Total Power Consumption (kW)')         # Y轴标签
plt.title('Energy Consumption vs Water Flow (Merkel ODE)') # 子图标题
plt.grid(True)                                     # 网格
plt.legend()                                       # 图例

plt.tight_layout()                                 # 自动调整间距
plt.savefig('14_Merkel_ODE_Coupled.png', dpi=300)  # 保存高清图片
plt.show()                                         # 弹窗展示

print("📸 耦合 ODE 物理仿真图像已保存为: 14_Merkel_ODE_Coupled.png") # 打印保存提示