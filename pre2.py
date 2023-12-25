import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt


'''city = ox.geocode_to_gdf(' New York')
ax = ox.project_gdf(city).plot()
_ = ax.axis('off')
plt.show()'''
#G = ox.graph_from_place('New York', network_type='drive')
#nx.write_graphml(G, "network.graphml")

G = ox.graph_from_bbox(40.918, 40.486, -73.7,-74.259, network_type='drive')


#G = ox.graph_from_bbox(40.918, 40.938, -73.7,-73.88, network_type='drive')
#G_projected = ox.project_graph(G)
ox.plot_graph(G)
#ox.save_graphml(G, filepath="network.graphml")

ox.save_graph_shapefile(G, filepath="Nyc_osm_data")









