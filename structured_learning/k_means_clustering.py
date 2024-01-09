

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Sample data of mythical creatures
creatures = {
    'Creature': ['Dragon', 'Unicorn', 'Griffin', 'Minotaur', 'Fairy', 'Hydra', 'Centaur', 'Elf', 'Troll', 'Mermaid'],
    'Size': [9, 4, 6, 7, 1, 8, 6, 3, 7, 5],
    'Magic Level': [8, 9, 5, 2, 10, 7, 4, 9, 1, 8],
    'Aggressiveness': [7, 1, 6, 6, 2, 8, 4, 2, 7, 3]
}
df = pd.DataFrame(creatures)

features = df[['Size', 'Magic Level', 'Aggressiveness']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
creature_clusters = kmeans.fit_predict(features_scaled)

# Add the cluster data to the original dataframe
df['Cluster'] = creature_clusters

plt.scatter(features_scaled[creature_clusters == 0, 0], features_scaled[creature_clusters == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(features_scaled[creature_clusters == 1, 0], features_scaled[creature_clusters == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(features_scaled[creature_clusters == 2, 0], features_scaled[creature_clusters == 2, 1], s=100, c='green', label='Cluster 3')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

# Annotating the data points
for i, txt in enumerate(df['Creature']):
    plt.annotate(txt, (features_scaled[i, 0], features_scaled[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Clusters of Mythical Creatures')
plt.legend()
plt.show()




