import numpy as np
import matplotlib.pyplot as plt

def my_fitting_tool(x_data, y_data, degree=2):
    """
    这是 june 的第一个专属拟合工具
    输入：x_data, y_data, degree
    输出：打印系数，显示并保存拟合图
    """
    # 1. 拟合
    coefs = np.polyfit(x_data, y_data, degree)
    print(f"Coefficients: {coefs}")

    # 2. 生成平滑曲线
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    y_smooth = np.polyval(coefs, x_smooth)

    # 3. 画图
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Fitted curve')
    plt.scatter(x_data, y_data, color='red', s=50, label='Original data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Fitting Tool (june)')
    plt.grid(True)
    plt.legend()

    # 4. 保存并显示（这两行必须有，才能结束画图流程）
    plt.savefig('fitted_plot.png', dpi=300, bbox_inches='tight')
    plt.show()