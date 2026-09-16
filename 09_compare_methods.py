"""
文件名: 09_compare_methods.py
功能: 对比两种优化算法（梯度下降 L-BFGS-B 和 遗传算法 GA），验证物理模型最优解的一致性。
作者: june
日期: 2026-09-17
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from scipy.optimize import minimize, differential_evolution  # 从 scipy.optimize 抽屉里，拿出 minimize 和 differential_evolution 两把工具

# --- 1. 物理模型（与 08 一致） ---
def total_power_physical(T_out):                   # 定义“物理总能耗”机器，原料是 T_out（出水温度）
    T_wb = 25.0                                    # 环境湿球温度，固定工况设为25度
    P_comp = 110 + 5 * (T_out - 30)                # 压缩机能耗：基准值110 + 斜率5×(实际温度-基准温度30)
    P_pump = 20.0 / (T_out - 29.4)                 # 水泵/风机能耗：反比函数，温度趋近29.4度时能耗无限大
    return P_comp + P_pump                         # 把两个能耗加起来，丢回去

# --- 2. 用 L-BFGS-B 梯度优化器求解 ---
result_lbfgs = minimize(                           # 让梯度优化器找最小值，结果装进 result_lbfgs 盒子里
    total_power_physical,                          # 目标函数
    x0=31.5,                                       # 初始猜测值：从31.5度开始找
    bounds=[(30.0, 33.0)],                         # 搜索范围：只能找30到33度之间的
    method='L-BFGS-B'                              # 方法：指定用 L-BFGS-B 这种算法
)
opt_temp_lbfgs = result_lbfgs.x[0]                 # 从盒子里掏出最优温度（x是数组，[0]取第一个值）
opt_power_lbfgs = result_lbfgs.fun                 # 从盒子里掏出最低能耗（fun是函数值）

# --- 3. 用遗传算法 (GA) 求解 ---
result_ga = differential_evolution(                # 让遗传算法找最小值，结果装进 result_ga 盒子里
    total_power_physical,                          # 目标函数
    bounds=[(30.0, 33.0)],                         # 搜索范围：30到33度
    strategy='best1bin',                           # 进化策略：一种变异规则
    maxiter=500,                                   # 最大迭代次数：最多繁衍500代
    popsize=20,                                    # 种群大小：每代派出20个个体去试
    tol=0.01,                                      # 收敛容差：变化小于0.01就收工
    seed=42                                        # 随机种子：保证每次跑结果一样
)
opt_temp_ga = result_ga.x[0]                       # 从盒子里掏出最优温度
opt_power_ga = result_ga.fun                       # 从盒子里掏出最低能耗

# --- 4. 画图对比 ---
T_range = np.linspace(30.0, 33.0, 100)             # 生成30到33度之间均匀分布的100个温度点
P_range = [total_power_physical(t) for t in T_range] # 列表推导式：100个温度挨个喂给机器，算出100个能耗值

plt.figure(figsize=(10, 6))                        # 设置画布大小，宽10，高6
# 物理模型曲线
plt.plot(T_range, P_range, 'b-', linewidth=2, label='Physical Model (Compressor + Pump)') # 画蓝色实线，线宽2

# L-BFGS-B 最优点（绿色菱形）
plt.scatter(opt_temp_lbfgs, opt_power_lbfgs, color='green', s=150, marker='D',             # 画绿色菱形（D代表Diamond菱形）
            label=f'L-BFGS-B Optimum ({opt_temp_lbfgs:.2f}°C, {opt_power_lbfgs:.2f} kW)')  # 标签带精确数值

# GA 最优点（红色星形）
plt.scatter(opt_temp_ga, opt_power_ga, color='red', s=200, marker='*',                     # 画红色星星（*代表星形）
            label=f'GA Optimum ({opt_temp_ga:.2f}°C, {opt_power_ga:.2f} kW)')              # 标签带精确数值

plt.xlabel('Cooling Tower Outlet Temperature (°C)')  # 给横坐标起名
plt.ylabel('Total Power Consumption (kW)')           # 给纵坐标起名
plt.title('Optimization Method Comparison on Physical Model (june)') # 图的标题
plt.grid(True)                                       # 打开网格线
plt.legend()                                         # 显示图例框
plt.savefig('08_Method_Comparison.png', dpi=300, bbox_inches='tight') # 保存高清图
plt.show()                                           # 弹窗展示

# --- 5. 终端输出对比结果 ---
print("\n" + "="*60)                               # 打印换行和60个等号
print("优化方法对比结果 (物理模型)")                 # 打印标题
print("="*60)                                      # 打印等号
print(f"L-BFGS-B : 最优温度 = {opt_temp_lbfgs:.4f} °C, 能耗 = {opt_power_lbfgs:.4f} kW") # 打印梯度法结果
print(f"GA        : 最优温度 = {opt_temp_ga:.4f} °C, 能耗 = {opt_power_ga:.4f} kW")      # 打印遗传法结果
print(f"两者差值  : 温度差 = {abs(opt_temp_lbfgs - opt_temp_ga):.4f} °C, 能耗差 = {abs(opt_power_lbfgs - opt_power_ga):.4f} kW") # 打印差值
print("="*60)                                      # 打印等号收尾