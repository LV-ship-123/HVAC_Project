"""
文件名: 11_merkel_ode.py
功能: 首次尝试用简化微分方程（ODE）模拟冷却塔水温变化（该版本存在物理漏洞，导致结果退化）。
作者: june
日期: 2026-09-17
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from scipy.integrate import odeint                 # 从 scipy 积分工具里，请出 odeint（常微分方程求解器）
from scipy.optimize import differential_evolution  # 从 scipy 优化工具里，请出遗传算法

# --- 1. 物理常数与模型参数 ---
Cp_w = 4.18       # 水的比热容 (kJ/(kg·°C))，表示1kg水升高1度需要吸收4.18千焦的热量
h_a = 50.0        # 容积传热系数 (kW/(m³·°C))，简化假设的固定值
A_v = 100.0       # 单位体积的换热面积 (m²/m³)，也是简化假设
V_tower = 2.0     # 填料体积 (m³)，冷却塔内部填料的空间大小

T_air_in = 28.0   # 进风干球温度 (°C)，也就是普通温度计测的空气温度
T_wb = 25.0       # 进风湿球温度 (°C)，这是冷却塔的理论冷却极限（水不可能比这个温度更低）

# --- 2. 简化的 Merkel 方程 (水温沿填料高度变化) ---
def merkel_ode(T_w, z, m_water, m_air):             # 定义微分方程函数，参数分别是：水温、高度、水流量、空气流量
    """
    简化的冷却塔微分方程
    dT_w/dz = -(h_a * A_v / (m_water * Cp_w)) * (T_w - T_wb)
    假设空气能瞬间带走热量，逼近湿球温度。
    这是最经典的“焓差驱动力”近似。
    """
    dT_dz = - (h_a * A_v / (m_water * Cp_w)) * (T_w - T_wb) # 核心公式：温度随高度的变化率
    return dT_dz                                     # 把变化率丢回去给求解器

# --- 3. 仿真函数：给定进水温度和水流量，计算出水温度和能耗 ---
def simulate_cooling_tower(m_water, m_air, T_water_in=37.0): # 定义仿真机器，输入水和空气流量，默认进水37度
    """
    输入：水流量 (kg/s)，空气流量 (kg/s)
    输出：出水温度 (°C)，风机能耗 (kW)，水泵能耗 (kW)
    """
    # 求解 ODE：从填料顶部（z=0）到底部（z=V_tower）
    z_range = np.linspace(0, V_tower, 100)           # 在0到2米之间，生成100个高度点
    T_initial = [T_water_in]                         # 初始状态：水温是37度
    
    try:                                             # try：尝试执行以下代码，如果出错就跳到except
        # 注意：odeint 要求参数以元组形式传递
        sol = odeint(merkel_ode, T_initial, z_range, args=(m_water, m_air)) # 解微分方程，求出100个高度对应的温度
        T_out = sol[-1][0]                           # 取出最后一个高度（塔底）的温度，也就是最终出水温度
    except:                                          # except：如果上面的计算崩了（比如除以0）
        # 如果数值求解发散，返回一个很大的惩罚值
        return 100.0, 1e6, 1e6                       # 返回100度和100万能耗，让优化器避开这种坏结果
    
    # 物理限制：出水温度不可能低于湿球温度
    if T_out < T_wb:                                 # 如果算出来的温度低于25度
        T_out = T_wb                                 # 强制把它锁死在25度（湿球温度）
    
    # --- 能耗计算 ---
    # 1. 水泵能耗：与流量成正比（扬程固定）
    P_pump = 0.8 * m_water                           # 假设每 kg/s 水耗电 0.8 kW
    
    # 2. 风机能耗：与空气流量立方成正比（真实风扇定律）
    P_fan = 0.15 * (m_air ** 3)                      # 风量越大，风机能耗呈立方级增加
    # 防止 m_air 太小导致数值不稳定，同时限制过大的风量
    if m_air < 0.1:                                  # 如果风量极小
        P_fan = 1e6                                  # 直接给一个极大的惩罚值（100万）
    
    P_total = P_pump + P_fan                         # 总能耗 = 水泵 + 风机
    return T_out, P_total                            # 返回出水温度和总能耗

# --- 4. 遗传算法优化：寻找最优的水流量和空气流量匹配 ---
def objective(x):                                    # 定义目标函数（遗传算法要优化的对象）
    m_water, m_air = x                               # 把输入的数组x，解包成两个值：水流量和空气流量
    # 假设进水温度固定在 37°C（典型工况）
    T_out, P_total = simulate_cooling_tower(m_water, m_air) # 调用上面的仿真机器
    
    # 加入惩罚项：如果出水温度高于 33°C，能耗指标很差，强行惩罚
    # 这里我们既要考虑能耗，又要满足出水温度在合理范围内（30~33°C）
    penalty = 0                                      # 初始化惩罚值为0
    if T_out > 33.0:                                 # 如果水温太高
        penalty = (T_out - 33.0) * 1000              # 超出33度的部分，乘以1000，作为巨额惩罚
    if T_out < 29.0:                                 # 如果水温太低（说明能耗极高）
        penalty = (29.0 - T_out) * 1000              # 低于29度的部分，乘以1000，作为惩罚
    
    return P_total + penalty                         # 返回“总能耗 + 惩罚值”，让优化器避开这些坑

print("🚀 开始优化物理模型（Merkel ODE 仿真）...")     # 终端打印提示
print("正在搜索最优的水流量与空气流量匹配...")        # 终端打印提示

# 搜索空间：水流量 2~15 kg/s，空气流量 0.5~8 kg/s
bounds = [(2.0, 15.0), (0.5, 8.0)]                   # 两个变量的搜索范围：水流量和水流量各有上下限

result = differential_evolution(                     # 调用遗传算法
    objective,                                       # 目标函数
    bounds,                                          # 搜索范围
    strategy='best1bin',                             # 进化策略
    maxiter=200,                                     # 最大迭代次数
    popsize=20,                                      # 种群大小
    tol=0.05,                                        # 收敛容差
    seed=42                                          # 随机种子，保证结果可复现
)

best_m_water, best_m_air = result.x                  # 从结果里，掏出最优的水流量和空气流量
best_T_out, best_P_total = simulate_cooling_tower(best_m_water, best_m_air) # 用这组最优参数，再算一次最终的出水和能耗

# --- 5. 输出结果 ---
print("\n" + "="*60)                                 # 打印分割线
print("✅ 遗传算法优化结果 (Merkel 动态物理模型)")       # 打印标题
print("="*60)                                        # 打印分割线
print(f"最优水流量   : {best_m_water:.3f} kg/s")       # 打印最优水流量
print(f"最优空气流量 : {best_m_air:.3f} kg/s")         # 打印最优空气流量
print(f"对应出水温度 : {best_T_out:.2f} °C")           # 打印出水温度
print(f"最低总能耗   : {best_P_total:.2f} kW")         # 打印最低能耗
print(f"收敛状态     : {result.success}")              # 打印是否成功
print("="*60)                                        # 打印分割线

# --- 6. 可视化：分析出水温度随水流量的变化关系（在最优风量下）---
m_water_range = np.linspace(2.0, 15.0, 50)            # 在水流量范围2到15之间，生成50个点
T_out_range = []                                     # 准备空列表，装出水温度
P_range = []                                         # 准备空列表，装能耗
for m_w in m_water_range:                            # 遍历50个水流量点
    T_out, P = simulate_cooling_tower(m_w, best_m_air) # 用固定的最优风量，算出不同水流量下的结果
    T_out_range.append(T_out)                        # 把温度追加到列表
    P_range.append(P)                                # 把能耗追加到列表

plt.figure(figsize=(12, 5))                          # 创建宽12高5的画布（双子图）

plt.subplot(1, 2, 1)                                 # 选择第1个子图（1行2列中的第1个）
plt.plot(m_water_range, T_out_range, 'b-', linewidth=2) # 画蓝色曲线：水流量 vs 出水温度
plt.scatter(best_m_water, best_T_out, color='red', s=100, marker='*', label='Optimum') # 画红色星星标记最优点
plt.xlabel('Water Flow Rate (kg/s)')                 # X轴标签
plt.ylabel('Outlet Temperature (°C)')                # Y轴标签
plt.title('Cooling Tower Performance vs Water Flow') # 子图标题
plt.grid(True)                                       # 打开网格线
plt.legend()                                         # 显示图例

plt.subplot(1, 2, 2)                                 # 选择第2个子图（1行2列中的第2个）
plt.plot(m_water_range, P_range, 'g-', linewidth=2)  # 画绿色曲线：水流量 vs 总能耗
plt.scatter(best_m_water, best_P_total, color='red', s=100, marker='*', label='Optimum') # 画红色星星标记最优点
plt.xlabel('Water Flow Rate (kg/s)')                 # X轴标签
plt.ylabel('Total Power Consumption (kW)')           # Y轴标签
plt.title('Energy Consumption vs Water Flow')        # 子图标题
plt.grid(True)                                       # 打开网格线
plt.legend()                                         # 显示图例

plt.tight_layout()                                   # 自动调整子图间距，防止重叠
plt.savefig('10_Merkel_ODE_Optimization.png', dpi=300) # 保存高清图片
plt.show()                                           # 弹窗展示

print("📸 物理仿真图像已保存为: 10_Merkel_ODE_Optimization.png") # 打印保存提示