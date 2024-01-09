import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample data of smartphones
smartphones = {
    'Model': ['PhoneA', 'PhoneB', 'PhoneC', 'PhoneD', 'PhoneE'],
    'Screen Size (inches)': [5.5, 6.1, 5.8, 6.4, 6.0],
    'Battery Life (hours)': [20, 23, 18, 25, 22],
    'Camera Quality (MP)': [12, 16, 14, 20, 18]
}
df = pd.DataFrame(smartphones)

features = df[['Screen Size (inches)', 'Battery Life (hours)', 'Camera Quality (MP)']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)  # Reducing to 2 dimensions
principal_components = pca.fit_transform(features_scaled)

# Create a DataFrame for the reduced data
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plotting the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.title('PCA of Smartphone Dataset')
plt.xlabel('Principal Component 1 - Mix of Size, Battery, and Camera')
plt.ylabel('Principal Component 2 - Mix of Size, Battery, and Camera')

# Annotating the points with model names
for i, txt in enumerate(df['Model']):
    plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]))

plt.show()
