from src import LabData,draw_three_plot_fft,draw_trajectory,get_time_domain_features,visible
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

#lab_dict = {'lab_misalignment', 'lab_normal'}
lab_dict = { 'lab_normal'}
# path = "DATA"

savefig_path = "lab1/"

data_slice=slice(15, 1500)



"""实验报告2批处理数据"""
path="DATA"
# lab_dic=['600rpm','700rpm','800rpm','1000rpm','1100rpm','1200rpm']
lab_dic=['9Hz','17Hz','20Hz','25Hz','29Hz']
data_dic=['电涡流传感器x','电涡流传感器y','加速度传感器']
datas=LabData()
# datas.get_file_path(path,['lab_normal'])
datas.get_file_path(path,['lab_misalignment'])
data_keys=datas.lab_file_path.keys()
# datas.show()
for val in lab_dic:
     print("begin to processing ",val)
     for key in data_keys:
             print("check ",key)
            #  print(data_keys)
             if val in key:
                        name=val
                        print(datas.lab_file_path[key])
                        print("begin to processing ",name)
                        data1=pd.read_excel(datas.lab_file_path[key][0])['Unnamed: 1'][data_slice]
                        data2=pd.read_excel(datas.lab_file_path[key][1])['Unnamed: 1'][data_slice]
                        # data3=pd.read_excel(datas.lab_file_path[key][2])['Unnamed: 1'][data_slice]
                        visible.draw_three_values_sub(data1, data2, name)
                        visible.sub_draw_three_plot_fft(data1, data2,name)
                        visible.draw_trajectory(data1, data2, name)
                        
            

print("all processing done")

# """用于交互测试"""
# if __name__ == "__main__":
   
#     # 进入交互模式
#     print("\n" + "="*50)
#     print("程序运行完成，进入交互模式")
#     print("你可以输入 Python 代码继续操作")
#     print("输入 'exit()' 或 'quit()' 退出")
#     print("="*50 + "\n")
    
    # import code
    # code.interact(local=locals())