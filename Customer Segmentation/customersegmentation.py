import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Data Generation ---
np.random.seed(42)  # For reproducibility
n_samples = 24574

customer_ids = np.arange(649000, 649000 + n_samples)
products_purchased = np.random.lognormal(mean=0.3, sigma=0.6, size=n_samples)
products_purchased = np.clip(products_purchased, 1, 13)
complains = np.random.binomial(1, 0.001051, size=n_samples)
money_spent = np.random.lognormal(mean=5, sigma=0.8, size=n_samples)
money_spent = money_spent * (191.5 / money_spent.mean())
money_spent = np.clip(money_spent, 0, 3132)

data = pd.DataFrame({
    'customer_id': customer_ids,
    'products_purchased': products_purchased,
    'complains': complains,
    'money_spent': money_spent
})

# --- Clustering and Analysis ---
features = ['products_purchased', 'complains', 'money_spent']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    if k > 1:
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

optimal_k = 4  # Example value; ideally, this would be based on the Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
cluster_centers_df.index.name = 'Cluster'

cluster_summary = data.groupby('cluster').agg({
    'customer_id': 'count',
    'products_purchased': 'mean',
    'complains': 'mean',
    'money_spent': 'mean'
}).rename(columns={'customer_id': 'count'})

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- Cluster Naming ---
def name_clusters(cluster_centers_df):
    cluster_names = []
    for i, row in cluster_centers_df.iterrows():
        if row['money_spent'] > 300:
            spending = "High-Spending"
        elif row['money_spent'] > 150:
            spending = "Medium-Spending"
        else:
            spending = "Low-Spending"
        
        if row['products_purchased'] > 3:
            products = "Diverse-Buyer"
        elif row['products_purchased'] > 1.5:
            products = "Multi-Buyer"
        else:
            products = "Single-Item"
        
        if row['complains'] > 0.1:
            satisfaction = "Dissatisfied"
        else:
            satisfaction = "Satisfied"
        
        cluster_names.append(f"Cluster {i}: {spending} {products} ({satisfaction})")
    return cluster_names

cluster_names = name_clusters(cluster_centers_df)

# --- GUI ---
root = tk.Tk()
root.title("Customer Segmentation Analysis")

# Notebook for Tabbed Interface
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=10, pady=10)

# --- Tab 1: Summary ---
tab_summary = ttk.Frame(notebook)
notebook.add(tab_summary, text="Summary")

# Summary Text Box
summary_text = scrolledtext.ScrolledText(tab_summary, height=20, width=80)
summary_text.pack(padx=10, pady=10, expand=True, fill="both")

# Function to add text to the summary
def add_summary_text(text):
    summary_text.insert(tk.END, text + "\n")

# --- Tab 2: Plots ---
tab_plots = ttk.Frame(notebook)
notebook.add(tab_plots, text="Plots")

# Frame to hold the plots
plots_frame = tk.Frame(tab_plots)
plots_frame.pack(expand=True, fill="both", padx=10, pady=10)

# --- Tab 3: Cluster Details ---
tab_clusters = ttk.Frame(notebook)
notebook.add(tab_clusters, text="Cluster Details")

# Cluster Details Text Box
cluster_text = scrolledtext.ScrolledText(tab_clusters, height=20, width=80)
cluster_text.pack(expand=True, fill="both", padx=10, pady=10)

# Function to add text to the cluster details
def add_cluster_text(text):
    cluster_text.insert(tk.END, text + "\n")

# --- Tab 4: Recommendations ---
tab_recommendations = ttk.Frame(notebook)
notebook.add(tab_recommendations, text="Recommendations")

# Recommendations Text Box
recommendations_text = scrolledtext.ScrolledText(tab_recommendations, height=20, width=80)
recommendations_text.pack(expand=True, fill="both", padx=10, pady=10)

# Function to add text to the recommendations
def add_recommendations_text(text):
    recommendations_text.insert(tk.END, text + "\n")

# --- Plotting Functions ---
def embed_plot(fig, parent):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# --- Create Plots ---
# Elbow Method Plot
fig_elbow, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(k_range, inertia, 'o-')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True)

axes[1].plot(k_range, silhouette_scores, 'o-')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Optimal k')
axes[1].grid(True)

fig_elbow.tight_layout()

# PCA Visualization Plot
fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster'], cmap='viridis', alpha=0.5)
ax_pca.set_xlabel('Principal Component 1')
ax_pca.set_ylabel('Principal Component 2')
ax_pca.set_title('Customer Segments Visualization using PCA')
plt.colorbar(scatter, label='Cluster', ax=ax_pca)
ax_pca.grid(True)

fig_pca.tight_layout()

# Box Plots and Cluster Sizes
fig_boxplots, axes_boxplots = plt.subplots(2, 2, figsize=(15, 10))
axes_boxplots = axes_boxplots.flatten()

for i, feature in enumerate(features):
    sns.boxplot(x='cluster', y=feature, data=data, ax=axes_boxplots[i])
    axes_boxplots[i].set_title(f'{feature} by Cluster')
    axes_boxplots[i].grid(True)

cluster_sizes = data['cluster'].value_counts().sort_index()
axes_boxplots[3].bar(cluster_sizes.index, cluster_sizes.values)
axes_boxplots[3].set_xlabel('Cluster')
axes_boxplots[3].set_ylabel('Number of Customers')
axes_boxplots[3].set_title('Cluster Sizes')
axes_boxplots[3].grid(True)

fig_boxplots.tight_layout()

# --- Add Plots to GUI ---
embed_plot(fig_elbow, plots_frame)
embed_plot(fig_pca, plots_frame)
embed_plot(fig_boxplots, plots_frame)

# --- Populate Text Boxes ---
add_summary_text("Dataset Summary:")
add_summary_text(str(data.describe()))
add_summary_text("\nCluster Centers:")
add_summary_text(str(cluster_centers_df))
add_summary_text("\nCluster Summary:")
add_summary_text(str(cluster_summary))

for i, name in enumerate(cluster_names):
    add_cluster_text(f"\n{name}:")
    add_cluster_text(str(cluster_summary.iloc[i]))

add_recommendations_text("Marketing Recommendations for Each Customer Segment:")
for name in cluster_names:
    cluster_id = int(name.split(':')[0].split()[-1])
    recommendations = f"\n{name}:\n"
    
    if "High-Spending" in name:
        recommendations += "- Focus on loyalty programs and premium services\n"
        recommendations += "- Offer early access to new products\n"
        recommendations += "- Provide personalized recommendations based on purchase history\n"
    
    if "Medium-Spending" in name:
        recommendations += "- Implement targeted upselling strategies\n"
        recommendations += "- Offer bundle deals to increase average order value\n"
        recommendations += "- Create mid-tier loyalty rewards\n"
    
    if "Low-Spending" in name:
        recommendations += "- Provide special discounts and promotions\n"
        recommendations += "- Implement email campaigns highlighting product value\n"
        recommendations += "- Use retargeting ads to increase engagement\n"
    
    if "Diverse-Buyer" in name:
        recommendations += "- Cross-sell complementary products\n"
        recommendations += "- Create category-specific promotions\n"
    
    if "Single-Item" in name:
        recommendations += "- Focus on product education to expand their interests\n"
        recommendations += "- Use 'customers also bought' recommendations\n"
    
    if "Dissatisfied" in name:
        recommendations += "- Implement proactive customer service outreach\n"
        recommendations += "- Offer satisfaction guarantees on future purchases\n"
        recommendations += "- Gather feedback to address common concerns\n"

    add_recommendations_text(recommendations)

# --- Run the GUI ---
root.mainloop()
