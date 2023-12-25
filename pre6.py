import pandas as pd
import osmnx as ox
import networkx as nx
import concurrent.futures
import time
from multiprocessing import Process
import time
import pandas as pd
import numpy as np
import multiprocessing
import warnings

from pandarallel import pandarallel  # 导入pandaralle

pandarallel.initialize()

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)

cup_num = multiprocessing.cpu_count()
#print(f"计算机cup核数：{cup_num}")
from pandarallel import pandarallel  # 导入pandaralle
pandarallel.initialize(nb_workers=6 ,progress_bar=True)  # 初始化该这个b..


def worker(row):

    orig = [row[4], row[3]]
    dest = [row[6], row[5]]
    origin_node = ox.nearest_nodes(G, orig[1], orig[0])
    destination_node = ox.nearest_nodes(G, dest[1], dest[0])
    try:
        route = nx.shortest_path(G, origin_node, destination_node, weight="length")
        route1_length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
    except:
        print("error")


def process_row(row):
    # 在这里执行行数据的处理操作
    print(f"Processing {row.Index}")

file_path='2014_yellow_10.csv'
df=pd.read_csv(file_path,nrows=100)
G = ox.load_graphml("network.graphml")
#df2=df.itertuples()
time_start=time.time()
print('0')
df['Sum'] = df.parallel_apply(worker, axis=1)
time_end=time.time()
print(time_end-time_start)
print(1)
print(len(df))


#for row in df.itertuples():
 #   if row.Index<=1000: