
import osmnx as ox
import networkx as nx
import pint
ureg = pint.UnitRegistry()
G = ox.load_graphml("network.graphml")
file_path=''

#ox.plot_graph(G)
#G = ox.graph_from_place('Piedmont, CA, USA', network_type='drive')
#origin = (33.307792, -111.894940)
pair=[-73.95780944824217,40.71775817871094,-73.96874237060547,40.677932739257805]
orig=(40.7177772,-73.927656860351563)
dest=(40.7410319519042969,-73.966835021972656)
orig=[pair[1],pair[0]]
dest=[pair[3],pair[2]]
print(1)
#G2 = ox.graph_from_bbox(40.79, 40.65, -73.70,-73.85, network_type='drive')
#G3 = ox.graph_from_bbox(40.75, 40.70, -73.75,-73.81, network_type='drive')
#ox.plot_graph(G2)
origin_node = ox.nearest_nodes(G, orig[1],orig[0])
#origin_node2 = ox.nearest_nodes(G3, orig[1],orig[0])
destination_node = ox.nearest_nodes(G, dest[1],dest[0])
#destination_node2 = ox.nearest_nodes(G3, dest[1],dest[0])
route = nx.shortest_path(G, origin_node, destination_node,weight="length")
#route2 = nx.shortest_path(G3, origin_node2, destination_node2,weight="length")
#fig, ax = ox.plot_graph_route(G2, route, route_linewidth=6, node_size=0, bgcolor='k')
#fig, ax = ox.plot_graph_route(G3, route2, route_linewidth=6, node_size=0, bgcolor='k')
print(route)
#print(route2)
route1_length = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
#route2_length = sum(ox.utils_graph.get_route_edge_attributes(G, route2, 'length'))
print(route1_length/1609.36)
for i in route:
    print(str(G.nodes[i]['y'])+','+str(G.nodes[i]['x']))
#print(route2_length/1609.36)

# ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')











