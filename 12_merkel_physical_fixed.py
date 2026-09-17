"""
文件名: 12_merkel_physical_fixed.py
功能: 尝试用 ε-NTU 效数法修正物理模型（该版本仍有漏洞，导致优化器压榨空气流量，未能解决水温被强行冷却的问题）。
作者: june
日期: 2026-09-17
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from scipy.optimize import differential_evolution  # 从 scipy 优化抽屉里，拿出 differential_evolution（遗传算法）

# --- 1. 物理常数与模型参数 (基于 ε-NTU 简化模型) ---
Cp_w = 4.18       # 水的比热容 (kJ/(kg·°C))，1kg水升高1度吸收4.18千焦热量
Cp_air = 1.005    # 空气的比热容 (kJ/(kg·°C))，1kg空气升高1度吸收1.005千焦热量
UA = 500.0        # 总传热系数 × 换热面积 (kW/°C)，反映冷却塔的整体换热能力

T_wb = 25.0       # 进风湿球温度 (°C)，也是理论冷却极限（水不可能比这个温度更低）

def simulate_cooling_tower(m_water, m_air, T_water_in=37.0): # 定义仿真机器，输入水流量、空气流量，默认进水37度
    """
    修正后的物理仿真：
    基于能量平衡和有限的换热能力（UA有限），空气流量不再是摆设。
    """
    # 1. 水侧放出的热量（如果冷却到湿球温度）
    Q_max = m_water * Cp_w * (T_water_in - T_wb)   # 水侧最多能放出的热量（理论上冷却到湿球温度）
    
    # 2. 空气侧能带走的热量（空气从湿球温度被加热到出水温度）
    # 为了防止空气温升无限，假设空气出口温度不能超过水入口温度
    # 空气侧最大吸热能力（假设空气被加热到水温）
    Q_air_max = m_air * Cp_air * (T_water_in - T_wb) # 空气侧最多能带走的热量
    
    # 3. 实际换热量：受限于 UA 换热能力和两侧流体的热容
    # 简化：实际换热量 = UA * (平均温差) * 时间，但这里用简化的“效数法”
    # 计算水侧热容率 C_water = m_water * Cp_w
    # 计算空气侧热容率 C_air = m_air * Cp_air
    C_min = min(m_water * Cp_w, m_air * Cp_air)    # 取水和空气两者热容率中较小的那个（min是取最小值）
    
    # 如果空气流量太小，C_min 会很小，换热效率会很高但总热量有限
    # 防止除零
    if C_min < 0.001:                              # 如果热容率极小（空气流量极小）
        return T_water_in, 1e6                     # 返回37度和100万惩罚，避免计算崩溃
    
    # 效数 (Effectiveness) 近似：假设逆流换热器
    # NTU = UA / C_min
    NTU = UA / C_min                               # 传热单元数（Number of Transfer Units）
    epsilon = 1 - np.exp(-NTU)                     # 效数（ε），这是传热学里计算换热器效率的经典公式
    
    # 实际换热量 = 效数 * 最大可能换热量
    Q_actual = epsilon * Q_max                     # 实际换热量，等于效数乘以水侧最大热量
    
    # 4. 计算实际出水温度
    T_out = T_water_in - Q_actual / (m_water * Cp_w) # 出水温度 = 进水温度 - (实际换热量 / 水侧热容率)
    
    # 物理安全限制：不能低于湿球温度
    if T_out < T_wb:                               # 如果算出来的温度低于25度
        T_out = T_wb                               # 强制锁死在25度
    
    # 5. 能耗计算 (与流量相关)
    # 水泵能耗：与水流量成正比（克服管道阻力）
    P_pump = 0.8 * m_water                         # 水泵能耗 = 0.8 × 水流量
    # 风机能耗：与空气流量立方成正比（风扇定律）
    P_fan = 0.15 * (m_air ** 3)                    # 风机能耗 = 0.15 × 空气流量的三次方
    
    # 惩罚项：确保出水温度在 30~35°C 之间（合理工况）
    penalty = 0                                    # 初始化惩罚值为0
    if T_out > 35.0:                               # 如果出水温度太高
        penalty += (T_out - 35.0) * 1000           # 超出35度的部分，乘以1000作为惩罚
    if T_out < 30.0:                               # 如果出水温度太低（说明水流量太大或空气流量太大）
        # 如果出水温度太低，说明水流量太大或空气流量太大，会消耗过多能量，给予惩罚
        penalty += (30.0 - T_out) * 100            # 低于30度的部分，乘以100作为惩罚
    
    P_total = P_pump + P_fan + penalty             # 总能耗 = 水泵 + 风机 + 惩罚
    
    return T_out, P_total                          # 返回出水温度和总能耗

# --- 2. 遗传算法优化 ---
def objective(x):                                  # 定义目标函数（遗传算法优化的对象）
    m_water, m_air = x                             # 解包，把数组 x 拆成水流量和空气流量
    T_out, P_total = simulate_cooling_tower(m_water, m_air) # 调用上面的仿真机器
    return P_total                                 # 返回总能耗（让优化器找最小值）

print("🚀 开始优化修正后的物理模型 (基于效数法)...")  # 打印提示
bounds = [(2.0, 15.0), (0.5, 8.0)]                 # 定义搜索范围：水流量2~15，空气流量0.5~8

result = differential_evolution(                   # 调用遗传算法
    objective,                                     # 目标函数
    bounds,                                        # 搜索范围
    strategy='best1bin',                           # 进化策略
    maxiter=150,                                   # 最大迭代次数
    popsize=20,                                    # 种群大小
    tol=0.05,                                      # 收敛容差
    seed=42                                        # 随机种子
)

best_m_water, best_m_air = result.x                # 从结果里掏出最优的水流量和空气流量
best_T_out, best_P_total = simulate_cooling_tower(best_m_water, best_m_air) # 用最优参数再算一次

print("\n" + "="*60)                               # 打印分割线
print("✅ 遗传算法优化结果 (修正后物理模型)")           # 打印标题
print("="*60)                                      # 打印分割线
print(f"最优水流量   : {best_m_water:.3f} kg/s")     # 打印最优水流量，保留3位小数
print(f"最优空气流量 : {best_m_air:.3f} kg/s")       # 打印最优空气流量，保留3位小数
print(f"对应出水温度 : {best_T_out:.2f} °C")         # 打印出水温度，保留2位小数
print(f"最低总能耗   : {best_P_total:.2f} kW")       # 打印最低能耗，保留2位小数
print(f"收敛状态     : {result.success}")            # 打印是否成功
print("="*60)                                      # 打印分割线

# --- 3. 可视化修正后的效果 ---
m_water_range = np.linspace(2.0, 15.0, 50)          # 生成50个水流量点
T_out_range = []                                   # 准备空列表，装出水温度
P_range = []                                       # 准备空列表，装能耗

# 固定最优空气流量，观察水流量变化的影响
print("\n正在绘制水流量对性能的影响...")              # 打印提示
for m_w in m_water_range:                          # 遍历50个水流量点
    T_out, P = simulate_cooling_tower(m_w, best_m_air) # 用固定的最优风量，算不同水流量下的结果
    T_out_range.append(T_out)                      # 把温度追加到列表
    P_range.append(P)                              # 把能耗追加到列表

plt.figure(figsize=(12, 5))                        # 创建宽12高5的画布（双子图）

plt.subplot(1, 2, 1)                               # 选择第1个子图
plt.plot(m_water_range, T_out_range, 'b-', linewidth=2) # 画蓝色曲线：水流量 vs 出水温度
plt.scatter(best_m_water, best_T_out, color='red', s=100, marker='*', label='Optimum') # 画红色星星标记最优点
plt.xlabel('Water Flow Rate (kg/s)')               # X轴标签
plt.ylabel('Outlet Temperature (°C)')              # Y轴标签
plt.title('Cooling Tower Performance (Fixed Air Flow)') # 子图标题
plt.grid(True)                                     # 打开网格线
plt.legend()                                       # 显示图例

plt.subplot(1, 2, 2)                               # 选择第2个子图
plt.plot(m_water_range, P_range, 'g-', linewidth=2) # 画绿色曲线：水流量 vs 总能耗
plt.scatter(best_m_water, best_P_total, color='red', s=100, marker='*', label='Optimum') # 画红色星星标记最优点
plt.xlabel('Water Flow Rate (kg/s)')               # X轴标签
plt.ylabel('Total Power Consumption (kW)')         # Y轴标签
plt.title('Energy Consumption vs Water Flow')      # 子图标题
plt.grid(True)                                     # 打开网格线
plt.legend()                                       # 显示图例

plt.tight_layout()                                 # 自动调整子图间距，防止重叠
plt.savefig('11_Merkel_Fixed_Optimization.png', dpi=300) # 保存高清图片
plt.show()                                         # 弹窗展示

print("📸 修正后的物理仿真图像已保存为: 11_Merkel_Fixed_Optimization.png") # 打印保存提示