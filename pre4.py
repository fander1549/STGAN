import osmnx as ox
import networkx as nx

#根据点和半径获取地图
wurster_hall = (37.870605, -122.254830)
one_mile = 1609  # meters
G = ox.graph_from_point(wurster_hall, dist=one_mile, network_type="drive")
fig, ax = ox.plot_graph(G, node_size=0)

#获取地点中的具体设施，可视化
G = ox.graph_from_place(
    "New York, New York, USA",
    retain_all=False,
    truncate_by_edge=True,
    simplify=True,
    custom_filter='["railway"~"subway"]',
)
fig, ax = ox.plot_graph(G, node_size=0, edge_color="w", edge_linewidth=0.2)

#获取设施的几何，可视化
place = "Civic Center, Los Angeles, California"
tags = {"building": True}
gdf = ox.geometries_from_place(place, tags)
fig, ax = ox.plot_footprints(gdf, figsize=(3, 3))

#根据地点获取路网，并可视化
G = ox.graph_from_place("Piedmont, California, USA", network_type="drive")
fig, ax = ox.plot_graph(G)

#将图结构的路网转变成geopandas格式的节点和边
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
gdf_nodes.head()

# 将geopandas格式的边和节点转变成Networkx的图
G2 = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=G.graph)

# 计算边的中心度，赋予边权重，并渲染出来
edge_centrality = nx.closeness_centrality(nx.line_graph(G))
nx.set_edge_attributes(G, edge_centrality, "edge_centrality")
edge_centrality = nx.closeness_centrality(nx.line_graph(G))
nx.set_edge_attributes(G, edge_centrality, "edge_centrality")

#最短路径搜索
#为网络添加速度和旅行时间
G = ox.speed.add_edge_speeds(G)
G = ox.speed.add_edge_travel_times(G)
# 将节点匹配到网络的最近路段上
orig = ox.distance.nearest_nodes(G, X=-122.245846, Y=37.828903)
dest = ox.distance.nearest_nodes(G, X=-122.215006, Y=37.812303)
# 生成最短路径并渲染
route = ox.shortest_path(G, orig, dest, weight="travel_time")
fig, ax = ox.plot_graph_route(G, route, node_size=0)
#路径长度计算
edge_lengths = ox.utils_graph.get_route_edge_attributes(G, route, "length")
round(sum(edge_lengths))