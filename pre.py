import pandas as pd
import os
import glob
# 读取CSV文件并将其加载到DataFrame中
#/Users/rfande/Downloads/2014
#E:\\BaiduNetdiskDownload\\2014
file_list = glob.glob(os.path.join("/Users/rfande/Downloads/2014", '*'))
count=0
for file_path  in file_list:

    #print(file_path)
    #df = pd.read_csv(file_path,header=0)
    #selected_cloumns=df.iloc[:,[1,2,4,5,6,9,10]]
    '''if 'green' in file_path:
        print(file_path)
        df = pd.read_csv(file_path,skiprows=3)
        selected_cloumns = df.iloc[:, [1, 2, 5,6 ,7 ,8 ,10 ]]#Green
        x=file_path[-6]
        y=file_path[-5]
        selected_cloumns.to_csv('2014_green_'+x+y+'.csv',index=False)
        print('2014_green_'+x+y+'.csv')'''
    if 'yellow' in file_path:
        print(file_path)
        df = pd.read_csv(file_path, skiprows=2)
        selected_cloumns=df.iloc[:,[1,2,4,5,6,9,10]]#yellow
        x = file_path[-6]
        y = file_path[-5]
        selected_cloumns.to_csv('2014_yellow_' + x + y + '.csv', index=False)
        print('2014_yellow_' + x + y + '.csv')

        #print(df.columns)
        #print(df.head())
        del df
    #elif count>=1 and count<12:
        #

        #print(file_path)
      #  df = pd.read_csv(file_path, header=0)
        # selected_cloumns=df.iloc[:,[1,2,4,5,6,9,10]]
       # selected_cloumns = df.iloc[:, [1, 2, 5, 6, 7, 8, 10]]
       # selected_cloumns.to_csv('2014_green_' + str(count + 1) + '.csv', index=False)
       # print('2014_green_' + str(count + 1) + '.csv')
        # print(df.columns)
        # print(df.head())
        #del df


