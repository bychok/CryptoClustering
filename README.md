![A vibrant and colorful image representing a data science project on cryptocurrency market analysis](colorful_crypto.webp)

# Cryptocurrency Market Data Analysis

This project aims to analyze and cluster cryptocurrencies based on their price change percentages over various time periods using Principal Component Analysis (PCA) and K-Means clustering.

## Table of Contents

1. [Import Required Libraries and Dependencies](#import-required-libraries-and-dependencies)
2. [Load and Display Data](#load-and-display-data)
3. [Generate Summary Statistics](#generate-summary-statistics)
4. [Prepare the Data](#prepare-the-data)
5. [Find the Best Value for k Using the Original Scaled DataFrame](#find-the-best-value-for-k-using-the-original-scaled-dataframe)
6. [Cluster Cryptocurrencies with K-means Using the Original Scaled Data](#cluster-cryptocurrencies-with-k-means-using-the-original-scaled-data)
7. [Optimize Clusters with Principal Component Analysis](#optimize-clusters-with-principal-component-analysis)
8. [Cluster Cryptocurrencies with K-means Using the PCA Data](#cluster-cryptocurrencies-with-k-means-using-the-pca-data)
9. [Determine the Weights of Each Feature on each Principal Component](#determine-the-weights-of-each-feature-on-each-principal-component)

## Import Required Libraries and Dependencies

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

## Load and Display Data

Load the cryptocurrency market data into a Pandas DataFrame and set the index to the `coin_id` column.

```python
market_data_df = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")
market_data_df.head(10)
```

## Generate Summary Statistics

```python
market_data_df.describe()
```

## Prepare the Data

Normalize the data using `StandardScaler` and create a DataFrame with the scaled data.

```python
scaled_market_data_df = StandardScaler().fit_transform(market_data_df)
scaled_market_data_df = pd.DataFrame(scaled_market_data_df, columns=market_data_df.columns)

crypto_names = market_data_df.index
scaled_market_data_df.set_index(crypto_names, inplace=True)

scaled_market_data_df.head()
```

## Find the Best Value for k Using the Original Scaled DataFrame

Create a list with the number of k-values to try, compute the inertia for each k-value, and plot the Elbow curve to identify the optimal k.

```python
k = list(range(1, 11))
inertia = []

for i in k:
    model = KMeans(n_clusters=i, random_state=1)
    model.fit(scaled_market_data_df)
    inertia.append(model.inertia_)

elbow_data = {"k": k, "inertia": inertia}
elbow_df = pd.DataFrame(elbow_data)

elbow_df.plot.line(x="k", y="inertia", title="Elbow Curve", ylabel='Inertia', xticks=k)
```

**Question:** What is the best value for k?

**Answer:** The optimal k-value is the point at which the graph forms an elbow: in this case, it is 4.

## Cluster Cryptocurrencies with K-means Using the Original Scaled Data

Initialize the K-Means model using the best value for k and fit the model using the scaled data.

```python
model = KMeans(n_clusters=4, random_state=1)
model.fit(scaled_market_data_df)

predictions = model.predict(scaled_market_data_df)

kmeans_predictions_df = scaled_market_data_df.copy()
kmeans_predictions_df['crypto_cluster'] = predictions

kmeans_predictions_df.head()
```

Create a scatter plot to visualize the clusters.

```python
kmeans_predictions_df.plot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    c="crypto_cluster",
    colormap="rainbow"
)
```

## Optimize Clusters with Principal Component Analysis

Create a PCA model instance, reduce the data to three principal components, and view the explained variance.

```python
pca_model = PCA(n_components=3, random_state=1)
pca_data = pca_model.fit_transform(scaled_market_data_df)

pca_model.explained_variance_ratio_
```

**Question:** What is the total explained variance of the three principal components?

**Answer:** The total explained variance of the three principal components is 89.50%.

## Cluster Cryptocurrencies with K-means Using the PCA Data

Repeat the clustering process using the PCA data.

```python
pca_market_data_df = pd.DataFrame(pca_data, columns=["PCA1", "PCA2", "PCA3"])
pca_market_data_df.set_index(crypto_names, inplace=True)

k = list(range(1, 11))
inertia = []

for i in k:
    k_model = KMeans(n_clusters=i, random_state=1)
    k_model.fit(pca_market_data_df)
    inertia.append(k_model.inertia_)

elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)

df_elbow.plot.line(x="k", y="inertia", title="Elbow Curve", ylabel='Inertia', xticks=k)
```

**Question:** What is the best value for k when using the PCA data?

**Answer:** The optimal k-value is the point at which the graph forms an elbow: in this case, it is 4.

**Question:** Does it differ from the best k value found using the original data?

**Answer:** No, the k-value found is the same as the original data.

Initialize the K-Means model using the best value for k and fit the model using the PCA data.

```python
model = KMeans(n_clusters=4, random_state=0)
model.fit(pca_market_data_df)

predictions = model.predict(pca_market_data_df)

pca_predictions_df = pca_market_data_df.copy()
pca_predictions_df['crypto_cluster'] = predictions

pca_predictions_df.head()

pca_predictions_df.plot.scatter(
    x="PCA1",
    y="PCA2",
    c="crypto_cluster",
    colormap="winter"
)
```

## Determine the Weights of Each Feature on each Principal Component

Use the columns from the original scaled DataFrame as the index to determine the weights.

```python
pd.DataFrame(pca_model.components_.T,
             columns=['PCA1', 'PCA2', 'PCA3'],
             index=scaled_market_data_df.columns)
```

**Question:** Which features have the strongest positive or negative influence on each component?

**Answer:**

- PCA1 - Strongest: `price_change_percentage_200d`
- PCA2 - Strongest: `price_change_percentage_30d`
- PCA3 - Strongest: `price_change_percentage_7d`
