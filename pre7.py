import pandas as pd
import osmnx as ox
import networkx as nx
file_path='2014_yellow_10.csv'
df=pd.read_csv(file_path,header=None)
count=len(df)
for row in df.itertuples():
    if row.Index<1000:
        print(row)
    count+=1

print(count)