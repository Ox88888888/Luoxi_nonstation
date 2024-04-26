import pandas as pd
import numpy as np

# 读取 Excel 文件
file_path = "C:/Users/10926/Desktop/712.xlsx"

df = pd.read_excel(file_path)

# 选择需要处理的列名，假设列名为 'ColumnName'
column_name = 'Z'

# 获取需要修改的正数部分
positive_values = df[df[column_name] == 1000]

# 生成从 1000 到 0 的整数列表，间隔为 2
new_values_positive = np.arange(1000, -1, -2)

# 将新的整数数值赋值给正数部分
df.loc[df[column_name] == 1000, column_name] = new_values_positive[:len(positive_values)]

# 将修改后的数据保存到新的 Excel 文件
output_file_path = 'E:/Qingfeng/712.xlsx'
df.to_excel(output_file_path, index=False)

print(f"已将处理后的数据保存到 {output_file_path} 文件中。")





