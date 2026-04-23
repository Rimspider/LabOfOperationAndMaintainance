import os
import pandas as pd
class LabData:
    def __init__(self):
        self.lab_paths = {}
        self.lab_subpaths = {}
        self.lab_files = {}
        self.lab_file_path={}
        self.dic={}

    """获取路径，并且可以通过lab_subpath键得到每个lab_subpath下的实验数据路径"""
    def get_file_path(self, base_path, lab_dict):
        self.__get_lab_paths__(base_path, lab_dict)
        self.__get_lab_subpath__()
        self.__get_lab_file_path__()

    """读取输入的路径"""
    def __get_lab_paths__(self, base_path, lab_dict):
        self.dic=lab_dict
        for lab in lab_dict:
            lab_path = os.path.normpath(os.path.join(base_path, lab))
            if os.path.exists(lab_path):
                self.lab_paths[lab] = lab_path
                self.lab_subpaths[lab]=[]
#                print(self.lab_paths)
            else:
                print(f"Warning: {lab} does not exist at {lab_path}")

    """获得输入路径下的子路径，并更改lab_subpaths"""
    def __get_lab_subpath__(self):
        for lab, path in self.lab_paths.items():
            subpaths = os.listdir(path)
            for subpath in subpaths:
                temp=os.path.normpath(os.path.join(path,subpath))
                if os.path.isdir(temp):
                    self.lab_subpaths[lab].append(temp)
#        print(self.lab_subpaths)

    """得到每个lab_path目录下的每个lab_subpath"""
    def __get_lab_file_path__(self):
        for lab in self.lab_subpaths:
            for subpath in self.lab_subpaths[lab]:
                self.lab_file_path[subpath]=[]
#                print(self.lab_subpaths[lab])
                files=os.listdir(subpath)
#                print(files)
                for file in files:
                    temp=os.path.normpath(os.path.join(subpath,file))
                    if (os.path.isfile(temp)) and ('vally' not in temp) and ('peak' not in temp):
                        self.lab_file_path[subpath].append(temp)
#        print(self.lab_file_path)
                    
    """将每个lab_subpath下的实验数据读出，通过lab_subpath键存储下"""
    def show(self):
        print("Lab paths:", self.lab_paths)
        print("Lab subpaths:", self.lab_subpaths)
        print("Lab file paths:", self.lab_file_path)

    """计算除了peak,valley外所有文件的行数，并且统计无法读取的文件数量"""
    def count_the_files_size(self):
        arr=[]
        count_error=0
        for lab in self.dic:
            for subpath in self.lab_subpaths[lab]:
                for file_path in self.lab_file_path[subpath]:
                    if "_peak" in file_path or "_vally" in file_path:
#                        print(f"Skipping: {file_path}")
                        continue
                    print(f"Processing: {file_path}")
                    if os.path.exists(file_path):
                        try:
                            count_data = pd.read_excel(file_path)
                            arr.append(len(count_data))
#                            print(f"Loaded {len(count_data)} rows from {os.path.basename(file_path)}")
                        except Exception as e:
#                            print(f"Error loading {file_path}: {e}")
                            count_error += 1
                    else:
                        count_error += 1
#                        print(f"File not found: {file_path}")

