"""
文件名: 13_merkel_true_physical.py
功能: 使用能量平衡法（min函数）修正物理模型，彻底解决空气流量未参与温度计算的问题。
作者: june
日期: 2026-09-17
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from scipy.optimize import differential_evolution  # 从 scipy 优化抽屉里，拿出 differential_evolution（遗传算法）

# --- 1. 物理常数 ---
Cp_w = 4.18        # 水的比热容 (kJ/(kg·°C))，1kg水升高1度吸收4.18千焦热量
Cp_air = 1.005     # 空气的比热容 (kJ/(kg·°C))，1kg空气升高1度吸收1.005千焦热量
T_wb = 25.0        # 进风湿球温度 (°C)，理论冷却极限
T_water_in = 37.0  # 进水温度 (°C)，冷却塔的入口水温固定为37度

def simulate_cooling_tower(m_water, m_air):        # 定义仿真机器，输入：水流量、空气流量
    """
    真正的能量平衡模型：
    空气流量直接决定换热能力。
    如果空气流量小，冷却效果差，出水温度接近进水温度。
    如果空气流量大，冷却效果好，出水温度接近湿球温度。
    """
    # 1. 计算水侧放热潜力（从37°C降到25°C）
    Q_water_max = m_water * Cp_w * (T_water_in - T_wb) # 水侧最多能放出多少热量
    
    # 2. 计算空气侧吸热潜力（从25°C升到37°C）
    Q_air_max = m_air * Cp_air * (T_water_in - T_wb)   # 空气侧最多能吸收多少热量
    
    # 3. 实际换热量 = 水侧潜力 和 空气侧潜力 的较小值（受限）
    # 这才是物理真相：谁少，谁就决定了换热量！(木桶效应，短板决定装水量)
    Q_actual = min(Q_water_max, Q_air_max)             # min()是Python内置函数，取括号里的最小值
    
    # 4. 如果空气流量极小，Q_actual 主要由空气侧决定
    # 计算实际出水温度：水温降低量 = Q_actual / (m_water * Cp_w)
    T_out = T_water_in - Q_actual / (m_water * Cp_w)   # 出水温度 = 进水温度 - (实际换热量 / 水侧热容率)
    
    # 物理安全限制
    if T_out < T_wb:                                   # 如果算出来的温度低于25度
        T_out = T_wb                                   # 强行锁死在25度
    if T_out > T_water_in:                             # 如果算出来的温度高于37度
        T_out = T_water_in                             # 强行锁死在37度
    
    # 5. 能耗计算 (真实风扇定律和水泵定律)
    # 水泵能耗：与流量成正比（克服管道阻力）
    P_pump = 0.8 * m_water                             # 水泵能耗 = 0.8 × 水流量
    
    # 风机能耗：与空气流量立方成正比（真实风扇定律）
    P_fan = 0.15 * (m_air ** 3)                        # 风机能耗 = 0.15 × 空气流量的三次方
    
    # 6. 惩罚项：如果出水温度太高（>33°C），说明冷却不足，施加高额惩罚
    penalty = 0                                        # 初始化惩罚值为0
    if T_out > 33.0:                                   # 如果出水温度大于33度
        penalty = (T_out - 33.0) * 1000                # 超出部分乘以1000作为惩罚
    if T_out < 29.0:                                   # 如果出水温度小于29度
        # 出水温度太低通常意味着能耗极高（空气或水流量过大），已经体现在能耗里了
        pass                                           # pass是“占位符”，表示什么都不做，跳过
    
    P_total = P_pump + P_fan + penalty                 # 总能耗 = 水泵 + 风机 + 惩罚
    return T_out, P_total                              # 返回出水温度和总能耗

# --- 2. 遗传算法优化 ---
def objective(x):                                      # 定义目标函数（遗传算法优化的对象）
    m_water, m_air = x                                 # 解包，把数组 x 拆成水流量和空气流量
    T_out, P_total = simulate_cooling_tower(m_water, m_air) # 调用上面的仿真机器
    return P_total                                     # 返回总能耗（让优化器找最小值）

print("🚀 开始优化真正的物理模型（能量平衡法）...")      # 终端打印提示
bounds = [(2.0, 15.0), (0.5, 8.0)]                     # 定义搜索范围：水流量2~15，空气流量0.5~8

result = differential_evolution(                       # 调用遗传算法
    objective,                                         # 目标函数
    bounds,                                            # 搜索范围
    strategy='best1bin',                               # 进化策略
    maxiter=200,                                       # 最大迭代次数
    popsize=20,                                        # 种群大小
    tol=0.05,                                          # 收敛容差
    seed=42                                            # 随机种子
)

best_m_water, best_m_air = result.x                    # 从结果里掏出最优的水流量和空气流量
best_T_out, best_P_total = simulate_cooling_tower(best_m_water, best_m_air) # 用最优参数再算一次最终结果

print("\n" + "="*60)                                   # 打印换行和60个等号
print("✅ 遗传算法优化结果 (真实能量平衡模型)")             # 打印标题
print("="*60)                                          # 打印等号
print(f"最优水流量   : {best_m_water:.3f} kg/s")         # 打印最优水流量，保留3位小数
print(f"最优空气流量 : {best_m_air:.3f} kg/s")           # 打印最优空气流量，保留3位小数
print(f"对应出水温度 : {best_T_out:.2f} °C")             # 打印出水温度，保留2位小数
print(f"最低总能耗   : {best_P_total:.2f} kW")           # 打印最低能耗，保留2位小数
print(f"收敛状态     : {result.success}")                # 打印是否成功
print("="*60)                                          # 打印等号

# --- 3. 可视化：固定最优空气流量，看水流量对温度和能耗的影响 ---
m_water_range = np.linspace(2.0, 15.0, 50)              # 生成50个水流量点
T_out_list = []                                        # 准备空列表，装出水温度
P_list = []                                            # 准备空列表，装能耗

print("\n正在绘制水流量对性能的影响...")                  # 打印提示
for m_w in m_water_range:                              # 遍历50个水流量点
    T_out, P = simulate_cooling_tower(m_w, best_m_air) # 用固定的最优风量，算不同水流量下的结果
    T_out_list.append(T_out)                           # 把温度追加到列表
    P_list.append(P)                                   # 把能耗追加到列表

plt.figure(figsize=(12, 5))                            # 创建宽12高5的画布（双子图）

plt.subplot(1, 2, 1)                                   # 选择第1个子图
plt.plot(m_water_range, T_out_list, 'b-', linewidth=2) # 画蓝色曲线：水流量 vs 出水温度
plt.scatter(best_m_water, best_T_out, color='red', s=100, marker='*', label='Optimum') # 画红色星星标记最优点
plt.xlabel('Water Flow Rate (kg/s)')                   # X轴标签
plt.ylabel('Outlet Temperature (°C)')                  # Y轴标签
plt.title('Cooling Tower Performance (True Physical Model)') # 子图标题
plt.grid(True)                                         # 打开网格线
plt.legend()                                           # 显示图例

plt.subplot(1, 2, 2)                                   # 选择第2个子图
plt.plot(m_water_range, P_list, 'g-', linewidth=2)     # 画绿色曲线：水流量 vs 总能耗
plt.scatter(best_m_water, best_P_total, color='red', s=100, marker='*', label='Optimum') # 画红色星星标记最优点
plt.xlabel('Water Flow Rate (kg/s)')                   # X轴标签
plt.ylabel('Total Power Consumption (kW)')             # Y轴标签
plt.title('Energy Consumption vs Water Flow')          # 子图标题
plt.grid(True)                                         # 打开网格线
plt.legend()                                           # 显示图例

plt.tight_layout()                                     # 自动调整子图间距，防止重叠
plt.savefig('12_True_Physical_Optimization.png', dpi=300) # 保存高清图片
plt.show()                                             # 弹窗展示

print("📸 真实物理仿真图像已保存为: 12_True_Physical_Optimization.png") # 打印保存提示