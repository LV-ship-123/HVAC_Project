"""
文件名: 02_my_toolkit.py
功能: 把拟合和画图的逻辑封装成一个函数，方便以后反复调用。
作者: june
日期: 2026-09-15
"""
import numpy as np            # 请数学工具包 numpy 来帮忙算数，给它起外号 np
import matplotlib.pyplot as plt # 请画图工具包 matplotlib 来帮忙画图，给它起外号 plt

# 定义一个函数：def就是"define"（定义）的缩写，my_fitting_tool 是我们给函数起的名字
def my_fitting_tool(x_data, y_data, degree=2) -> None:
    """
    这是 june 的第一个专属拟合工具
    输入：x_data, y_data, degree
    输出：打印系数，显示并保存拟合图
    """
    # 1. 拟合（让数学工具算出那条二次曲线）
    coefs = np.polyfit(x_data, y_data, degree) # 让 np 用多项式拟合，找出系数 a, b, c
    print(f"Coefficients: {coefs}")            # 把算出来的系数打印在终端上给你看一眼
    
    # 2. 生成平滑曲线（算出一百个点来连线，这样线条才不会像折线一样尖锐）
    x_smooth = np.linspace(min(x_data), max(x_data), 100) # 在最小x和最大x之间，均匀生成100个数
    y_smooth = np.polyval(coefs, x_smooth)               # 把这100个x代入公式，算出对应的100个y值
    
    # 3. 画图（把算出来的数据变成图形）
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Fitted curve') # 画蓝色的拟合曲线，宽度为2，标签名叫"拟合曲线"
    plt.scatter(x_data, y_data, color='red', s=50, label='Original data') # 画红色散点图，点大小为50，标签名叫"原始数据"
    plt.xlabel('X')                           # 给横坐标起名，就叫 X
    plt.ylabel('Y')                           # 给纵坐标起名，就叫 Y
    plt.title('Fitting Tool (june)')          # 给整张图起个标题
    plt.grid(True)                            # 打开背景网格线，方便肉眼估算
    plt.legend()                              # 显示图例框，把刚才起好名字的"曲线"和"点"汇总显示出来
    
    # 4. 保存并显示（这两行必须有，才能结束画图流程）
    plt.savefig('fitted_plot.png', dpi=300, bbox_inches='tight') # 把图保存成名为 fitted_plot.png 的高清图片
    plt.show()                                # 把画好的图在屏幕上弹出来