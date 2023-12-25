import \
    pandas as pd
import os


import glob
# 读取CSV文件并将其加载到DataFrame中
#/Users/rfande/Downloads/2014

#file_list = glob.glob(os.path.join("E:\\BaiduNetdiskDownload\\2014", '*'))
count=1000
file_path='/Users/rfande/Downloads/2014/green_tripdata_2014-10.csv'
print(file_path)
df = pd.read_csv(file_path,skiprows=3)
#selected_cloumns=df.iloc[:,[1,2,4,5,6,9,10]]#yellow
selected_cloumns = df.iloc[:, [1, 2, 5,6 ,7 ,8 ,10 ]]#GREEN
selected_cloumns.to_csv('2014_green_'+str(count+1)+'.csv',index=False)
#print('2014_green_'+str(count+1)+'.csv')
#print(df.columns)
#print(df.head())

        #del df


