"""
文件名: 14_merkel_with_penalty.py
功能: 在能量平衡模型基础上，引入工程惩罚项，迫使优化器避开低流量等不安全运行区间。
作者: june
日期: 2026-09-18
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from scipy.optimize import differential_evolution  # 从 scipy 优化抽屉里，拿出 differential_evolution（遗传算法）

# --- 1. 物理常数 ---
Cp_w = 4.18        # 水的比热容 (kJ/(kg·°C))
Cp_air = 1.005     # 空气的比热容 (kJ/(kg·°C))
T_wb = 25.0        # 进风湿球温度 (°C)
T_water_in = 37.0  # 进水温度 (°C)

def simulate_cooling_tower(m_water, m_air):        # 定义仿真机器，输入：水流量、空气流量
    """
    真实能量平衡 + 工程可行域惩罚（让低流量变得昂贵）
    """
    # 1. 计算换热
    Q_water_max = m_water * Cp_w * (T_water_in - T_wb) # 水侧最多能放出多少热量
    Q_air_max = m_air * Cp_air * (T_water_in - T_wb)   # 空气侧最多能吸收多少热量
    Q_actual = min(Q_water_max, Q_air_max)             # 实际换热量：谁少谁说了算（木桶效应）
    T_out = T_water_in - Q_actual / (m_water * Cp_w)   # 出水温度 = 进水温度 - 降温幅度
    
    # 物理安全限制
    if T_out < T_wb:                                   # 如果算出来的温度低于25度
        T_out = T_wb                                   # 强行锁死在25度
    if T_out > T_water_in:                             # 如果算出来的温度高于37度
        T_out = T_water_in                             # 强行锁死在37度
    
    # 2. 能耗计算
    P_pump = 0.8 * m_water                             # 水泵能耗 = 0.8 × 水流量
    P_fan = 0.15 * (m_air ** 3)                        # 风机能耗 = 0.15 × 空气流量三次方
    
    # 3. 工程惩罚（新加入的“驱赶”机制）
    # 冷却塔在低流量时性能急剧恶化，用分段函数模拟
    penalty = 0                                        # 初始化惩罚值为0
    
    # 惩罚1：水流量太低（< 4 kg/s），导致冷却效果不可靠
    if m_water < 4.0:                                  # 如果水流量低于4
        penalty += 200.0 * (4.0 - m_water) ** 2        # 惩罚值累加：200乘以(4减去水流量)的平方。越界越痛！
    
    # 惩罚2：出水温度过高（> 33.5°C），系统效率下降
    if T_out > 33.5:                                   # 如果出水温度超过33.5度
        penalty += 100.0 * (T_out - 33.5) ** 2         # 罚！
    
    # 惩罚3：出水温度过低（< 28°C），意味着水量太大，浪费能量
    if T_out < 28.0:                                   # 如果出水温度低于28度
        penalty += 50.0 * (28.0 - T_out) ** 2          # 罚！
    
    P_total = P_pump + P_fan + penalty                 # 总能耗 = 水泵 + 风机 + 惩罚
    return T_out, P_total                              # 返回出水温度和总能耗

# --- 2. 遗传算法优化 ---
def objective(x):                                      # 定义目标函数
    m_water, m_air = x                                 # 解包数组x
    _, P_total = simulate_cooling_tower(m_water, m_air)# 调用仿真机器，忽略温度，只拿能耗
    return P_total                                     # 返回总能耗，让优化器寻找最小值

print("🚀 开始优化带工程惩罚的物理模型...")              # 打印提示
bounds = [(2.0, 15.0), (0.5, 8.0)]                     # 搜索范围：水流量2~15，空气流量0.5~8

result = differential_evolution(                       # 调用遗传算法
    objective,                                         # 目标函数
    bounds,                                            # 搜索范围
    strategy='best1bin',                               # 进化策略
    maxiter=200,                                       # 最大迭代次数
    popsize=20,                                        # 种群大小
    tol=0.05,                                          # 收敛容差
    seed=42                                            # 随机种子
)

best_m_water, best_m_air = result.x                    # 提取最优参数
best_T_out, best_P_total = simulate_cooling_tower(best_m_water, best_m_air) # 用最优参数再算一次

print("\n" + "="*60)                                   # 打印换行和60个等号
print("✅ 遗传算法优化结果 (带工程惩罚)")                   # 打印标题
print("="*60)                                          # 打印等号
print(f"最优水流量   : {best_m_water:.3f} kg/s")         # 打印最优水流量，保留3位小数
print(f"最优空气流量 : {best_m_air:.3f} kg/s")           # 打印最优空气流量，保留3位小数
print(f"对应出水温度 : {best_T_out:.2f} °C")             # 打印出水温度，保留2位小数
print(f"最低总能耗   : {best_P_total:.2f} kW")           # 打印最低能耗，保留2位小数
print(f"收敛状态     : {result.success}")                # 打印是否成功
print("="*60)                                          # 打印等号

# --- 3. 可视化 ---
m_water_range = np.linspace(2.0, 15.0, 100)            # 生成100个水流量点
T_out_list = []                                        # 空列表装温度
P_list = []                                            # 空列表装能耗

print("正在绘制水流量对性能的影响...")                    # 打印提示
for m_w in m_water_range:                              # 遍历水流量点
    T_out, P = simulate_cooling_tower(m_w, best_m_air) # 用固定最优风量算结果
    T_out_list.append(T_out)                           # 追加温度
    P_list.append(P)                                   # 追加能耗

plt.figure(figsize=(12, 5))                            # 创建画布

plt.subplot(1, 2, 1)                                   # 第1个子图
plt.plot(m_water_range, T_out_list, 'b-', linewidth=2) # 画蓝色曲线
plt.scatter(best_m_water, best_T_out, color='red', s=100, marker='*', label='Optimum') # 画红色星星
plt.axvline(4.0, color='gray', linestyle='--', alpha=0.7, label='Min reliable flow (4 kg/s)') # 画灰色的虚线参考线，位置在x=4.0
plt.xlabel('Water Flow Rate (kg/s)')                   # X轴标签
plt.ylabel('Outlet Temperature (°C)')                  # Y轴标签
plt.title('Cooling Tower Performance (with Penalty)') # 子图标题
plt.grid(True)                                         # 网格
plt.legend()                                           # 图例

plt.subplot(1, 2, 2)                                   # 第2个子图
plt.plot(m_water_range, P_list, 'g-', linewidth=2)     # 画绿色曲线
plt.scatter(best_m_water, best_P_total, color='red', s=100, marker='*', label='Optimum') # 画红色星星
plt.axvline(4.0, color='gray', linestyle='--', alpha=0.7) # 画灰色虚线参考线
plt.xlabel('Water Flow Rate (kg/s)')                   # X轴标签
plt.ylabel('Total Power Consumption (kW)')             # Y轴标签
plt.title('Energy Consumption vs Water Flow (with Penalty)') # 子图标题
plt.grid(True)                                         # 网格
plt.legend()                                           # 图例

plt.tight_layout()                                     # 自动调整间距
plt.savefig('13_Merkel_With_Penalty.png', dpi=300)     # 保存高清图片
plt.show()                                             # 弹窗展示

print("📸 带惩罚的物理仿真图像已保存为: 13_Merkel_With_Penalty.png") # 打印保存提示