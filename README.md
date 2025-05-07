# K-Means Clustering on Mall Customers Dataset

This project performs unsupervised learning using the K-Means clustering algorithm to segment customers based on their features. The dataset used is the `Mall_Customers.csv`, which contains demographic and spending information of mall customers.

## 🧰 Tools and Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib
- PCA (for visualization)
- Silhouette Score (for evaluation)

## 📁 Dataset
The dataset contains the following columns:
- `CustomerID`
- `Genre`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

## 🚀 Steps Performed

### 1. Load and Prepare Data
- Read the CSV file using `pandas`.
- Drop non-numeric or irrelevant columns like `CustomerID` and `Genre`.
- Standardize the data using `StandardScaler`.

### 2. Dimensionality Reduction (Optional)
- Reduce data to 2D using **PCA** for easier visualization.

### 3. Determine Optimal K
- Use the **Elbow Method** to plot Within-Cluster-Sum-of-Squares (WCSS) for `K = 1` to `10`.

### 4. Apply K-Means
- Fit the K-Means model with the chosen number of clusters (e.g., K=5).
- Assign each customer to a cluster.

### 5. Visualize Clusters
- Visualize the customer segments in 2D (PCA-reduced) space with color-coded clusters.

### 6. Evaluate Model
- Use **Silhouette Score** to measure how well the data has been clustered.

## 📊 Sample Output
- **Elbow Curve** to identify optimal `K`
- **Scatter Plot** of PCA-reduced clusters
- **Silhouette Score** printed in console


