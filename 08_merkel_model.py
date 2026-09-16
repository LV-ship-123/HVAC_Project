"""
文件名: 08_merkel_model.py
功能: 建立第一个物理权衡模型（压缩机能耗 vs 水泵能耗），并用遗传算法求出最低能耗对应的出水温度。
作者: june
日期: 2026-09-17
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from scipy.optimize import differential_evolution  # 从优化工具箱里请出差分进化（遗传算法的高级变种）

# --- 1. 物理模型（模拟压缩机能耗 vs 水泵/风机能耗的权衡） ---
def total_power_physical(T_out):                   # 定义“物理总能耗”机器，原料是 T_out（出水温度）
    """
    输入: 冷却塔出水设定温度 (°C)
    输出: 系统总能耗 (kW)
    模拟了真实的工程权衡:
    - 温度越低，制冷机省电，但水泵耗电剧增（反比关系）
    - 温度越高，水泵省电，但制冷机耗电增加
    """
    T_wb = 25.0                                    # 环境湿球温度，固定工况设为25度（冷却塔的极限温度）

    # 1. 压缩机能耗：随出水温度升高而线性增加（冷凝温度升高）
    # 30°C 时为 120 kW，33°C 时为 112.5 kW（实际上随温度升高而升高，修正一下逻辑）
    # 正确的物理逻辑：出水温度越低，压缩机越省电。
    # 设定：30°C 时压缩机能耗最低(110kW)，33°C 时能耗最高(125kW)
    P_comp = 110 + 5 * (T_out - 30)                # 压缩机能耗公式：基础值110 + 斜率5×(实际温度-基准温度30)

    # 2. 水泵/风机能耗：出水温度越低，需要越大的水流量/风量，功耗呈指数/反比上升
    # 这里用“反比例”模拟物理极限：趋近 29.5°C 时，功耗疯狂增加
    # 同时保证在 33°C 时，这部分功耗降到最低
    P_pump = 20.0 / (T_out - 29.4)                 # 水泵能耗公式：20除以(温度-29.4)，当温度趋近29.4时，分母趋近0，能耗无限大
    
    # 总能耗 = 压缩机 + 水泵/风机
    P_total = P_comp + P_pump                      # 把上面算出的两个能耗加起来
    return P_total                                 # 把这台机器算出来的总能耗丢出去

# --- 2. 用遗传算法找全局最优解 ---
print("正在用遗传算法搜索物理模型的最优出水温度...")  # 在终端打印一句话，通知你开始计算了
result = differential_evolution(                   # 让遗传算法去自动寻找最小值，结果装进 result 盒子里
    total_power_physical,                          # 目标函数：算这个物理总能耗的最小值
    bounds=[(30.0, 33.0)],                         # 搜索区间：只能在30到33度之间找
    strategy='best1bin',                           # 进化策略：一种变异规则，默认即可
    maxiter=500,                                   # 最大迭代次数：最多繁衍500代
    popsize=20,                                    # 种群大小：每一代派出20个个体去试
    tol=0.01,                                      # 收敛容差：如果变化小于0.01就收工
    seed=42                                        # 随机种子：保证每次跑结果一样
)

# --- 3. 输出结果 ---
print("\n" + "="*50)                               # 打印一个空行，紧接着打印50个等号，作为分割线
print("遗传算法优化结果 (物理权衡模型)")             # 打印标题
print("="*50)                                      # 再打印50个等号
print(f"✅ 最优出水温度: {result.x[0]:.4f} °C")    # 打印：最优温度（取result盒子里的x的第1个值，保留4位小数）
print(f"✅ 最小总能耗: {result.fun:.4f} kW")       # 打印：最低能耗（取result盒子里的fun属性，保留4位小数）
print(f"✅ 收敛状态: {result.success}")            # 打印：是否成功（True或False）
print(f"✅ 函数评估次数: {result.nfev}")            # 打印：算法算了几次函数（nfe评估次数）
print("="*50 + "\n")                               # 打印50个等号加一个换行，用于收尾

# --- 4. 画出物理模型的完整曲线（标出最优点） ---
T_range = np.linspace(30.0, 33.0, 100)             # 生成从30到33度、均匀分布的100个温度点
P_range = [total_power_physical(t) for t in T_range] # 列表推导式：把100个温度挨个喂给机器，算出100个能耗值

plt.figure(figsize=(8, 5))                         # 设置画布大小，宽8，高5
plt.plot(T_range, P_range, 'b-', linewidth=2, label='Physical Model (Compressor + Pump)') # 用蓝色实线画能耗曲线
plt.scatter(result.x[0], result.fun, color='red', s=150, marker='*',                       # 在最优温度点上画一颗红色大星星
            label=f'GA Optimum ({result.x[0]:.2f}°C, {result.fun:.2f} kW)')                 # 星星的标签包含精确数值
plt.xlabel('Cooling Tower Outlet Temperature (°C)')  # 给横坐标起名：冷却塔出水温度
plt.ylabel('Total Power Consumption (kW)')           # 给纵坐标起名：总能耗
plt.title('Physical Merkel-like Model Optimization (june)') # 给图起个标题
plt.grid(True)                                       # 打开背景网格线
plt.legend()                                         # 显示图例（告诉大家蓝线是啥、红星星是啥）
plt.savefig('07_Physical_Optimum.png', dpi=300, bbox_inches='tight') # 保存高清图（名字叫07_Physical_Optimum.png）
plt.show()                                           # 弹窗把图画出来
print("📸 物理模型图像已保存为: 07_Physical_Optimum.png") # 在终端通知图已经保存好了