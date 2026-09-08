import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# --- 1. 物理模型（模拟压缩机能耗 vs 水泵/风机能耗的权衡） ---
def total_power_physical(T_out):
    """
    输入：冷却塔出水设定温度 (°C)
    输出：系统总能耗 (kW)
    模拟了真实的工程权衡：
    - 温度越低，制冷机省电，但水泵耗电剧增（反比关系）
    - 温度越高，水泵省电，但制冷机耗电增加
    """
    T_wb = 25.0  # 环境湿球温度（固定工况）

    # 1. 压缩机功耗：随出水温度升高而线性增加（冷凝温度升高）
    # 30°C 时为 120 kW，33°C 时为 112.5 kW（实际上随温度升高而升高，修正一下逻辑）
    # 正确的物理逻辑：出水温度越低，压缩机越省电。
    # 设定：30°C 时压缩机能耗最低(110kW)，33°C 时能耗最高(125kW)
    P_comp = 110 + 5 * (T_out - 30)  

    # 2. 水泵/风机功耗：出水温度越低，需要越大的水流量/风量，功耗呈指数/反比上升
    # 这里用“反比例”模拟物理极限：趋近 29.5°C 时，功耗疯狂增加
    # 同时保证在 33°C 时，这部分功耗降到最低
    P_pump = 20.0 / (T_out - 29.4)  

    # 总能耗 = 压缩机 + 水泵/风机
    P_total = P_comp + P_pump
    return P_total

# --- 2. 用遗传算法找全局最优解 ---
print("正在用遗传算法搜索物理模型的最优出水温度...")
result = differential_evolution(
    total_power_physical,
    bounds=[(30.0, 33.0)],   # 搜索区间
    strategy='best1bin',
    maxiter=500,
    popsize=20,
    tol=0.01,
    seed=42
)

# --- 3. 输出结果 ---
print("\n" + "="*50)
print("遗传算法优化结果 (物理权衡模型)")
print("="*50)
print(f"✅ 最优出水温度: {result.x[0]:.4f} °C")
print(f"✅ 最小总能耗: {result.fun:.4f} kW")
print(f"✅ 收敛状态: {result.success}")
print(f"✅ 函数评估次数: {result.nfev}")
print("="*50 + "\n")

# --- 4. 画出物理模型的完整曲线（标出最优点）---
T_range = np.linspace(30.0, 33.0, 100)
P_range = [total_power_physical(t) for t in T_range]

plt.figure(figsize=(8, 5))
plt.plot(T_range, P_range, 'b-', linewidth=2, label='Physical Model (Compressor + Pump)')
plt.scatter(result.x[0], result.fun, color='red', s=150, marker='*', 
            label=f'GA Optimum ({result.x[0]:.2f}°C, {result.fun:.2f} kW)')
plt.xlabel('Cooling Tower Outlet Temperature (°C)')
plt.ylabel('Total Power Consumption (kW)')
plt.title('Physical Merkel-like Model Optimization (june)')
plt.grid(True)
plt.legend()
plt.savefig('07_Physical_Optimum.png', dpi=300, bbox_inches='tight')
plt.show()

print("📸 物理模型图像已保存为: 07_Physical_Optimum.png")