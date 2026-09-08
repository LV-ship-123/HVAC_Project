import importlib

# 动态导入以数字开头的模块
my_toolkit = importlib.import_module("02_my_toolkit")

# 然后像正常使用一样调用函数
T = [30, 31, 32, 33]
P = [108, 109.5, 110.5, 111.5]

my_toolkit.my_fitting_tool(T, P)