"""
文件名: utils.py
功能: 存放公共函数（如通用拟合绘图、优化器配置、ODE求解参数等），供其他脚本调用。
作者: june
日期: 2026-09-15
"""
# 第一步：导入需要用到的工具箱（就像把锤子、螺丝刀准备好）
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp

# ================= 1. 拟合绘图工具 =================
def my_fitting_tool(x_data, y_data, degree=2):
    """
    功能: 封装多项式拟合和绘图逻辑，输入数据，输出拟合系数。
    输入: x_data（横坐标数据）, y_data（纵坐标数据）, degree（拟合次数，默认2次）
    输出: 拟合系数 coefs
    """
    # 1. 让 numpy 帮我们找出穿过这些点的二次多项式系数（a, b, c）
    coefs = np.polyfit(x_data, y_data, degree)
    
    # 2. 生成 100 个平滑的 x 点（为了画图时曲线更光滑，而不是折线）
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    
    # 3. 把这些 x 带入公式，算出对应的 y 值
    y_smooth = np.polyval(coefs, x_smooth)
    
    # 4. 开始画图（画线、画点、贴标签）
    plt.plot(x_smooth, y_smooth, 'b-', label='Fitted curve')
    plt.scatter(x_data, y_data, color='red', label='Original data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 5. 把系数返回给调用者
    return coefs


# ================= 2. 优化器配置工具 =================
def get_lbfgsb_config():
    """
    功能: 返回 L-BFGS-B 梯度优化器的配置参数
    说明: 这是一个字典，包含初始值、范围等，供 minimize 函数调用。
    """
    # 创建一个字典（类似于一个带标签的盒子），里面装着参数
    config = {
        'x0': 31.5,                # 初始猜测值（从31.5度开始找）
        'bounds': [(30.0, 33.0)],   # 搜索范围：[30度, 33度]
        'method': 'L-BFGS-B'        # 指定优化方法
    }
    return config

def get_ga_config():
    """
    功能: 返回遗传算法 (GA) 的配置参数
    说明: 供 differential_evolution 函数调用。
    """
    # 同样是一个字典，装着遗传算法的专属参数
    config = {
        'bounds': [(30.0, 33.0)],   # 搜索范围
        'strategy': 'best1bin',     # 进化策略
        'maxiter': 500,             # 最大迭代次数
        'popsize': 20,              # 种群大小
        'tol': 0.01,                # 收敛容差
        'seed': 42                  # 随机种子（保证每次结果一样）
    }
    return config


# ================= 3. Merkel ODE 求解公共参数 =================
def get_ode_params():
    """
    功能: 返回 Merkel ODE 求解所需的物理参数和求解器配置
    说明: 供 solve_ivp 函数调用。
    """
    # 把物理参数和求解器设置打包成一个字典
    params = {
        # 物理参数
        'T_in': 37.0,        # 进水温度 (°C)
        'T_wb': 25.0,        # 环境湿球温度 (°C)
        'H': 1.0,            # 填料高度 (m)
        'K': 0.8,            # 容积传热系数 (常数，简化设定)
        # 求解器配置
        't_span': [0, 1.0],  # 求解范围：从 0 到 1.0
        'rtol': 1e-4,        # 相对误差容忍度（越小越精准，但越慢）
        'atol': 1e-6         # 绝对误差容忍度
    }
    return params


# ================= 4. 独立测试区 =================
# 这一段是专门用来测试的。只要在终端运行 python utils.py，就会执行这里的代码。
if __name__ == "__main__":
    print("================ 开始独立测试 utils.py ================")
    
    # 测试 1：优化器配置
    print("\n[1] 测试优化器配置：")
    # 调用上面定义的函数，把结果装进变量里
    lbfgsb_cfg = get_lbfgsb_config()
    # 打印出结果，f"" 是格式化字符串，可以把变量塞进文本里
    print(f"L-BFGS-B 的初始值: {lbfgsb_cfg['x0']}，搜索范围: {lbfgsb_cfg['bounds']}")
    ga_cfg = get_ga_config()
    print(f"GA 算法的最大迭代次数: {ga_cfg['maxiter']}，种群大小: {ga_cfg['popsize']}")
    
    # 测试 2：ODE 求解配置
    print("\n[2] 测试 ODE 求解配置：")
    ode_cfg = get_ode_params()
    print(f"进水温度: {ode_cfg['T_in']}°C，湿球温度: {ode_cfg['T_wb']}°C")
    print(f"ODE 求解容差 rtol={ode_cfg['rtol']}, atol={ode_cfg['atol']}")

        # 测试 3：画图测试（验证 my_fitting_tool 能否独立画图）
    print("\n[3] 测试画图功能：")
    
    # 准备两组测试数据（这里直接用论文图8a的4个点）
    T_test = [30, 31, 32, 33]
    P_test = [108, 109.5, 110.5, 111.5]
    
    # 打印提示，告诉用户图马上弹出来
    print("正在生成拟合图...")
    
    # 调用我们封装好的画图函数
    my_fitting_tool(T_test, P_test)
    
    # 图关掉之后，打印这句提示，表示流程结束
    print("画图测试完成！")
    
    print("\n================ 测试全部通过！utils.py 运行正常！ ================")