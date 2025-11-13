import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

if not os.path.exists('result'):
    os.makedirs('result')

# -----------------------------
# 1. Uploading data
# -----------------------------
file_path = 'data/walmart_sales.csv'
data = pd.read_csv(file_path)

print("Data succesfully downloaded")
print(data.head())

print("\nInformation about data:")
print(data.info())

print("\nOmission in data")
print(data.isnull().sum())

# -----------------------------
# 2. Preparing features for the graph
# -----------------------------
# Average values of indicators for stores
store_features = data.groupby('Store')[['Temperature','Fuel_Price','CPI','Unemployment']].mean()
print("\nAverage values of indicators for stores:")
print(store_features.head())

# Threshold for connections
threshold = {
    'Temperature': 5,   
    'Fuel_Price': 0.3,  
    'CPI': 5,           
    'Unemployment': 1   
}

# -----------------------------
# 3. Graph construction
# -----------------------------
G = nx.Graph()
stores = store_features.index.tolist()
G.add_nodes_from(stores)

for i in range(len(stores)):
    for j in range(i+1, len(stores)):
        s1 = store_features.loc[stores[i]]
        s2 = store_features.loc[stores[j]]
        if (abs(s1.Temperature - s2.Temperature) < threshold['Temperature'] and
            abs(s1.Fuel_Price - s2.Fuel_Price) < threshold['Fuel_Price'] and
            abs(s1.CPI - s2.CPI) < threshold['CPI'] and
            abs(s1.Unemployment - s2.Unemployment) < threshold['Unemployment']):
            G.add_edge(stores[i], stores[j])

print(f"\nNumber of nodes: {G.number_of_nodes()}")
print(f"Number of connections: {G.number_of_edges()}")

# -----------------------------
# 4. Graph visualization
# -----------------------------
plt.figure(figsize=(10,8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
plt.title("Walmart store graph by similarity of economic conditions")
plt.savefig('result/graph_similarity.png') 
plt.show()



# -----------------------------
# 5. Graph metrics
# -----------------------------
# Degree (степень узла)
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, 'degree')

# Betweenness centrality (посредническая центральность)
betweenness_dict = nx.betweenness_centrality(G)
nx.set_node_attributes(G, betweenness_dict, 'betweenness')

# Clustering coefficient (коэффициент кластеризации)
clustering_dict = nx.clustering(G)
nx.set_node_attributes(G, clustering_dict, 'clustering')

# Print top 5 stores by metrics
print("\nTop 5 stores by degree:")
print(sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5])

print("\nTop 5 stores by betweenness centrality:")
print(sorted(betweenness_dict.items(), key=lambda x: x[1], reverse=True)[:5])

print("\nTop 5 stores by clustering coefficient:")
print(sorted(clustering_dict.items(), key=lambda x: x[1], reverse=True)[:5])

# -----------------------------
# 6. Community detection
# -----------------------------
from networkx.algorithms.community import greedy_modularity_communities

communities = list(greedy_modularity_communities(G))
community_dict = {}
for i, com in enumerate(communities):
    for node in com:
        community_dict[node] = i
nx.set_node_attributes(G, community_dict, 'community')

print(f"\nNumber of detected communities: {len(communities)}")

# -----------------------------
# 7. Graph visualization with communities
# -----------------------------
plt.figure(figsize=(12,10))
pos = nx.spring_layout(G, seed=42)
colors = [community_dict[node] for node in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.tab20, node_size=500)
edges = nx.draw_networkx_edges(G, pos, alpha=0.5)
labels = nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Walmart store graph with communities")
plt.savefig('result/graph_communities.png')
plt.show()



# Degree distribution histogram (Figure 7)
plt.figure(figsize=(8,6))
plt.hist(list(degree_dict.values()), bins=range(1, max(degree_dict.values())+2), color='skyblue', edgecolor='black')
plt.xlabel('Degree')
plt.ylabel('Number of Stores')
plt.title('Degree Distribution of Walmart Store Graph')
plt.savefig('result/degree_distribution.png')
plt.show()


# Figure 6
top_nodes = sorted(betweenness_dict.items(), key=lambda x: x[1], reverse=True)[:5]
top_nodes_list = [node for node, _ in top_nodes]

plt.figure(figsize=(12,10))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=400)
nx.draw_networkx_nodes(G, pos, nodelist=top_nodes_list, node_color='red', node_size=700)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Top-5 Key Nodes by Betweenness Centrality")
plt.savefig('result/top_nodes.png')
plt.show()


# Figure 9
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
pos_spring = nx.spring_layout(G, seed=42)
nx.draw(G, pos_spring, with_labels=True, node_color='skyblue', node_size=500)
plt.title("Spring Layout")

plt.subplot(1,2,2)
pos_circular = nx.circular_layout(G)
nx.draw(G, pos_circular, with_labels=True, node_color='skyblue', node_size=500)
plt.title("Circular Layout")
plt.savefig('result/layout_comparison.png')
plt.show()
