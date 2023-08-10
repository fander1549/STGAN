import pandas as pd
import os
import glob
# 读取CSV文件并将其加载到DataFrame中
file_list = glob.glob(os.path.join("E:\\BaiduNetdiskDownload\\2014", '*'))
count=0
for file_path  in file_list:
    if count<1:
        print(file_path)
        df = pd.read_csv(file_path,header=0)
        #selected_cloumns=df.iloc[:,[1,2,4,5,6,9,10]]
        selected_cloumns = df.iloc[:, [1, 2, 5,6 ,7 ,8 ,10 ]]
        selected_cloumns.to_csv('2014_green_'+str(count+1)+'.csv',index=False)
        print('2014_green_'+str(count+1)+'.csv')
        #print(df.columns)
        #print(df.head())
        del df
    elif count>=1 and count<12:
        print(file_path)
        df = pd.read_csv(file_path, header=0)
        # selected_cloumns=df.iloc[:,[1,2,4,5,6,9,10]]
        selected_cloumns = df.iloc[:, [1, 2, 5, 6, 7, 8, 10]]
        selected_cloumns.to_csv('2014_green_' + str(count + 1) + '.csv', index=False)
        print('2014_green_' + str(count + 1) + '.csv')
        # print(df.columns)
        # print(df.head())
        del df
    count+=1
