"""
文件名: 10_robustness_analysis.py
功能: 通过给物理模型参数施加随机扰动，验证最优温度的稳定性（鲁棒性分析）。
作者: june
日期: 2026-09-17
"""
import numpy as np                                 # 请数学工具箱 numpy 进来，起外号叫 np
import matplotlib.pyplot as plt                    # 请画图工具箱 matplotlib.pyplot 进来，起外号叫 plt
from scipy.optimize import differential_evolution  # 从 scipy.optimize 抽屉里，拿出 differential_evolution（遗传算法）

# 物理模型（原始版本）
def total_power_physical(T_out, slope=5.0, pump_strength=20.0):  # 定义机器，多加了两个可调节的参数（斜率和泵强度）
    P_comp = 110 + slope * (T_out - 30)            # 压缩机能耗：把原本写死的5换成了变量slope
    P_pump = pump_strength / (T_out - 29.4)        # 水泵能耗：把原本写死的20换成了变量pump_strength
    return P_comp + P_pump                         # 加总，把结果丢出去

# 鲁棒性测试设置
n_trials = 50                                      # 设定跑50次扰动实验
optimal_temps = []                                 # 准备一个空列表，用来收集每次实验得到的最优温度

print("🔬 开始鲁棒性测试：对物理模型参数进行随机扰动...") # 在终端打印提示，告诉你开始算了
for i in range(n_trials):                          # for循环，让 i 从 0 变到 49，一共循环50次
    # 对压缩机斜率（±20%）和水泵强度（±20%）施加随机干扰
    perturbed_slope = 5.0 * np.random.uniform(0.8, 1.2)   # 在0.8到1.2之间随机生成一个数，乘以基准斜率5.0
    perturbed_pump = 20.0 * np.random.uniform(0.8, 1.2)   # 在0.8到1.2之间随机生成一个数，乘以基准强度20.0

    # 用遗传算法搜索最优点
    result = differential_evolution(               # 让遗传算法寻找最小值，结果装进result盒子里
        lambda x: total_power_physical(x[0], slope=perturbed_slope, pump_strength=perturbed_pump), # 临时打包一个函数，用上刚才随机生成的参数
        bounds=[(30.0, 33.0)],                     # 搜索区间：只能在30到33度之间找
        strategy='best1bin',                       # 进化策略：一种变异规则
        maxiter=500,                               # 最大迭代次数：500代
        popsize=15,                                # 种群大小：每代15个测试点
        tol=0.01,                                  # 收敛容差
        seed=None                                  # 随机种子设为None，意味着每次运行结果都随机（这才是真正的测试）
    )
    optimal_temps.append(result.x[0])              # 把本次找到的最优温度，追加到列表里
    print(f"  第 {i+1:2d} 次: 扰动后最优温度 = {result.x[0]:.4f} °C") # 打印本次结果（i+1是因为序号从0开始，要加1才是1到50次）

# --- 统计分析 ---
temps = np.array(optimal_temps)                    # 把收集了50个温度的列表，转换成数学数组，方便算统计量
mean_temp = np.mean(temps)                         # 计算平均值（mean）
std_temp = np.std(temps)                           # 计算标准差（standard deviation，衡量数据的波动程度）
min_temp = np.min(temps)                           # 找出最小值（min）
max_temp = np.max(temps)                           # 找出最大值（max）

print("\n" + "="*60)                               # 打印换行和60个等号
print("📊 鲁棒性测试结果 (50次随机扰动)")             # 打印标题
print("="*60)                                      # 打印等号
print(f"平均最优温度: {mean_temp:.4f} °C")           # 打印平均值
print(f"标准差:       {std_temp:.4f} °C")            # 打印标准差
print(f"最小值:       {min_temp:.4f} °C")            # 打印最小值
print(f"最大值:       {max_temp:.4f} °C")            # 打印最大值
print(f"波动范围:     [{min_temp:.4f}, {max_temp:.4f}]") # 打印范围
print(f"95%置信区间:  [{mean_temp - 1.96*std_temp:.4f}, {mean_temp + 1.96*std_temp:.4f}]") # 打印95%置信区间（统计学公式）
print("="*60)                                      # 打印等号

# --- 绘制分布直方图 ---
plt.figure(figsize=(10, 5))                        # 准备画布，宽10高5
plt.hist(temps, bins=20, edgecolor='black', alpha=0.7, color='skyblue') # 画直方图，分20个柱子，黑边，半透明，天蓝色
plt.axvline(mean_temp, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_temp:.2f}°C') # 画一条红色虚线代表平均值
plt.axvline(31.4, color='green', linestyle='-', linewidth=1.5, label='Original Optimum (31.40°C)') # 画一条绿色实线代表原始最优值
plt.xlabel('Optimal Temperature (°C)')             # 横坐标标签
plt.ylabel('Frequency')                            # 纵坐标标签（频率）
plt.title('Robustness Test: Optimal Temperature Distribution under Parameter Perturbation (june)') # 图的标题
plt.legend()                                       # 显示图例
plt.grid(True, alpha=0.3)                          # 打开网格线，透明度0.3
plt.savefig('09_Robustness_Distribution.png', dpi=300, bbox_inches='tight') # 保存高清图片
plt.show()                                         # 弹窗展示
print("📸 鲁棒性分布图已保存为: 09_Robustness_Distribution.png") # 打印保存提示