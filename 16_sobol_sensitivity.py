"""
文件名: 16_sobol_sensitivity.py
功能: 使用 Sobol 全局敏感性分析方法，量化各参数对最优出水温度的影响权重。
作者: june
日期: 2026-09-18
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from SALib.sample import saltelli                  # 从 SALib 的 sample（采样）模块里，请出 saltelli（一种高效的采样方法）
from SALib.analyze import sobol                    # 从 SALib 的 analyze（分析）模块里，请出 sobol（Sobol 敏感性分析算法）
from scipy.optimize import differential_evolution  # 从 scipy 优化工具箱里，请出 differential_evolution（遗传算法）

# 1. 物理模型
def total_power_physical(T_out, slope, pump_strength):  # 定义“物理总能耗”机器，原料是出水温度、压缩机斜率、泵强度
    P_comp = 110 + slope * (T_out - 30)            # 压缩机能耗 = 110 + 斜率 × (温度 - 30)
    P_pump = pump_strength / (T_out - 29.4)        # 水泵能耗 = 泵强度 / (温度 - 29.4)
    return P_comp + P_pump                         # 总能耗 = 压缩机能耗 + 水泵能耗，把结果丢回去

# 2. 目标函数（优化器用它寻找最优温度）
def objective(x):                                  # 定义目标函数，原料是数组 x
    T_out = x[0]                                   # 从数组 x 里取出第 1 个值（即出水温度）
    # 这里的 slope 和 pump_strength 由 Sobol 采样决定
    return total_power_physical(T_out, current_slope, current_pump) # 调用上面的物理模型，把当前循环里 Sobol 采样的全局变量传进去

# 3. 定义敏感性分析的输入参数范围（这里是重点！）
# 给压缩机斜率一个范围(4.0~6.0)，给泵强度一个范围(15.0~25.0)
problem = {                                        # 定义一个字典（用大括号 {} 包裹），起名叫 problem
    'num_vars': 2,                                 # 'num_vars' 是键，2 是值，表示有 2 个变量
    'names': ['slope', 'pump_strength'],           # 'names' 是键，后面列表是 2 个变量的名字
    'bounds': [[4.0, 6.0], [15.0, 25.0]]           # 'bounds' 是键，后面是每个变量对应的上下限范围
}

# 4. 生成 Sobol 采样序列
# N 是采样基数，数值越大越精确，但计算越慢。先设 64 试试水
param_values = saltelli.sample(problem, 64)        # 调用 saltelli 工具，根据 problem 里的范围，生成 64 组采样点
print(f"总共要进行 {len(param_values)} 次优化计算，请耐心等待...") # 打印提示，告诉你总共要算多少组（f"..." 是格式化字符串）

# 5. 跑一遍所有的采样点，记录对应的最优温度
Y = np.zeros([param_values.shape[0]])              # 创建一个全为0的数组 Y，长度与采样点数量一致，用来装结果

for i in range(len(param_values)):                 # for 循环，i 从 0 到采样点的总数，挨个遍历
    # 把当前采样的参数传给目标函数
    current_slope = param_values[i, 0]             # 取出第 i 组的第 1 个值（斜率），赋值给 current_slope
    current_pump = param_values[i, 1]              # 取出第 i 组的第 2 个值（泵强度），赋值给 current_pump
    
    # 给定当前物理参数，用遗传算法找最低能耗对应的温度
    result = differential_evolution(               # 调用遗传算法
        objective,                                 # 目标函数
        bounds=[(30.0, 33.0)],                     # 搜索区间：只能在30到33度之间找
        maxiter=200,                               # 最大迭代次数200
        popsize=15,                                # 种群大小15
        tol=0.01,                                  # 收敛容差0.01
        seed=42                                    # 随机种子42
    )
    Y[i] = result.x[0]                             # 从 result 盒子里掏出最优温度，存入数组 Y 的第 i 个位置
    if (i+1) % 50 == 0:                            # 如果 i+1 能被 50 整除（即每到第50次）
        print(f"  已完成 {i+1}/{len(param_values)} 次计算...") # 打印进度提示

# 6. 进行 Sobol 敏感性分析
print("\n正在计算 Sobol 敏感度指数...")              # 打印提示
Si = sobol.analyze(problem, Y)                     # 调用 sobol 工具，根据 problem 和结果 Y 算出敏感性指数

# 7. 输出结果
print("\n" + "="*60)                               # 打印换行和60个等号
print("📊 Sobol 敏感性分析结果 (对最优温度的影响)")   # 打印标题（📊 是打出来的Emoji表情）
print("="*60)                                      # 打印等号
print(f"参数 slope (压缩机斜率) 的 S1 (一阶灵敏度): {Si['S1'][0]:.4f}")     # 打印斜率的 S1 指数
print(f"参数 pump_strength (泵强度) 的 S1 (一阶灵敏度): {Si['S1'][1]:.4f}") # 打印泵强度的 S1 指数
print(f"参数 slope 的 ST (总灵敏度): {Si['ST'][0]:.4f}")                    # 打印斜率的 ST 指数
print(f"参数 pump_strength 的 ST (总灵敏度): {Si['ST'][1]:.4f}")            # 打印泵强度的 ST 指数
print("="*60)                                      # 打印等号
print("【大白话解释】S1越大，说明这个参数单独对温度影响越大；ST越大，说明它与其他参数交叉影响也大。") # 打印解释

# 8. 画柱状图展示灵敏度
fig, ax = plt.subplots(figsize=(8, 5))             # 创建一块画布，宽8高5。fig 是画布对象，ax 是坐标轴对象
x = np.arange(len(problem['names']))               # 生成一个数列：[0, 1]，用于横坐标的位置
width = 0.35                                       # 设置柱子的宽度为0.35

ax.bar(x - width/2, Si['S1'], width, label='First-order (S1)', color='skyblue')# 画蓝色的 S1 柱子，位置在 x 左偏半个宽度
ax.bar(x + width/2, Si['ST'], width, label='Total-order (ST)', color='salmon')  # 画橙红色的 ST 柱子，位置在 x 右偏半个宽度

ax.set_ylabel('Sensitivity Index')                 # 给 Y 轴起名：敏感度指数
ax.set_title('Sobol Sensitivity Analysis on Optimal Temperature (june)') # 给图起标题
ax.set_xticks(x)                                   # 设置 X 轴的刻度位置
ax.set_xticklabels(problem['names'])               # 把 X 轴的刻度标签换成变量名（slope, pump_strength）
ax.legend()                                        # 显示图例框
plt.grid(True, alpha=0.3)                          # 打开网格，透明度0.3
plt.tight_layout()                                 # 自动调整子图间距，防止重叠
plt.savefig('15_Sobol_Sensitivity.png', dpi=300)   # 保存高清图片
plt.show()                                         # 弹窗把图展示出来
print("📸 敏感性分析图已保存为: 15_Sobol_Sensitivity.png") # 打印保存提示（📸 也是Emoji表情）