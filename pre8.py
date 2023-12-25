import matplotlib.pyplot as plt

# 经纬度坐标数据
coordinates = [
    (40.7180046, -73.9282209), (40.718179, -73.9282933), (40.7182133, -73.9281501), (40.7188256, -73.9284413),
    (40.7172161, -73.9347055), (40.7226071, -73.9371073), (40.7234043, -73.9340271), (40.7244078, -73.934468),
    (40.725476, -73.93491), (40.725907, -73.935398), (40.7264162, -73.9360815), (40.7274928, -73.9362535),
    (40.7281, -73.938189), (40.7275817, -73.9438743), (40.7274038, -73.945735), (40.7273769, -73.9458951),
    (40.730884, -73.947895), (40.7307945, -73.9481716), (40.7307721, -73.9482893), (40.7304718, -73.9514256),
    (40.7308639, -73.9516316), (40.7311028, -73.9517005), (40.7339983, -73.9521783), (40.7347036, -73.9523388),
    (40.7349832, -73.9524468), (40.7354416, -73.9525534), (40.737648, -73.9529152), (40.7378509, -73.9529291),
    (40.738055, -73.9529129), (40.7384923, -73.9527873), (40.7423262, -73.951497), (40.7428799, -73.9513403),
    (40.7432648, -73.9512578), (40.7431897, -73.9514078), (40.743183, -73.9515361), (40.7430643, -73.9517582),
    (40.742981, -73.9518215), (40.7426909, -73.9523902), (40.7424742, -73.9527543), (40.7419294, -73.9537796),
    (40.7420235, -73.9543437), (40.7413435, -73.9545788), (40.7417859, -73.9568975), (40.7418606, -73.956926),
    (40.7419265, -73.9569898), (40.7420759, -73.9573194), (40.7421339, -73.9576145), (40.7421415, -73.9577237),
    (40.7420917, -73.9580943), (40.7421228, -73.958591), (40.7421891, -73.9589108), (40.7391262, -73.9599472),
    (40.7390828, -73.9600015), (40.7390213, -73.9601298), (40.7385911, -73.9612519), (40.7385901, -73.9613317),
    (40.7386263, -73.961385), (40.7394141, -73.9618403), (40.7394672, -73.961848)

]

# 提取纬度和经度
latitudes = [coord[0] for coord in coordinates]
longitudes = [coord[1] for coord in coordinates]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制经纬度坐标点
plt.scatter(longitudes, latitudes, c='blue', marker='o', s=50)

# 添加标题和轴标签
plt.title('Plotting Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 显示图表
plt.show()