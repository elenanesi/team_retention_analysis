import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the input dataset
df = pd.read_excel("team_retention.xlsx")

# ranking columns to include in the analysis
# TODO: aggiungere salario normalizzato per paese tra rank columns?

rank_columns = [
    'Rank of Salary in reasons why to leave your job',
    'Rank of Toxic Environment in reasons why to leave your job',
    'Rank of Lack of growth opportunities in reasons why to leave your job',
    'Rank of Lack of work-life balance in reasons why to leave your job',
    'Rank of Lack of organizational structure within the company in reasons why to leave your job',
    'Rank of Lack of attention to ethics, sustainability, equity in reasons why to leave your job',
    'Rank of Lack of confidence in managers in reasons why to leave your job',
    'Rank of Feeling disrespected by managers in reasons why to leave your job'
]

data_for_clustering = df[rank_columns]

# Impute NaN values with the mean for each ranking column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_imputed = imputer.fit_transform(data_for_clustering)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)


# Determined optimal number of clusters as 3 w Elbow method
optimal_clusters = 3

# cluster analysis
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(data_scaled)

# Print the first few rows to check cluster assignment
print(df.head())

# Analyzing clusters - averaging the rankings for each cluster
cluster_analysis = df.groupby('Cluster')[rank_columns].mean()
print(cluster_analysis)

# Handle salary ranges and open-ended values correctly (including 'k')
def salary_range_to_midpoint(range_str):
    if 'over' in range_str:
        lower_bound = float(range_str.split(' ')[1].replace('k', '')) * 1000
        return lower_bound + 5000  # Adjust as needed, assuming +5k as a placeholder
    elif '-' in range_str:
        lower, upper = range_str.split(' - ')
        lower_bound = float(lower.replace('k', '')) * 1000
        upper_bound = float(upper.replace('k', '')) * 1000
        return (lower_bound + upper_bound) / 2
    else:
        return float(range_str.replace('k', '')) * 1000


# Clean 'Age' 
df['Age'] = df['Age'].apply(lambda x: age_range_to_midpoint(x) if isinstance(x, str) else x)

# Clean 'Annual salary' 
df['Annual salary'] = df['Annual salary'].apply(lambda x: salary_range_to_midpoint(x) if isinstance(x, str) else x)


# Print the characteristics of each cluster
for cluster_num in range(optimal_clusters):
    print(f"\nCluster {cluster_num} characteristics:\n")
    
    # Filter for the current cluster
    cluster_data = df[df['Cluster'] == cluster_num]
    
    # Print basic demographics and job characteristics
    print(f"Total members in this cluster: {len(cluster_data)}")
    
    # Average age - No need for a custom calculation since 'Age' is already numeric
    print(f"Average Age: {cluster_data['Age'].mean():.2f}")
    
    # Gender distribution
    print("Gender distribution:")
    print(cluster_data['Gender'].value_counts(normalize=True) * 100)
    
    # Seniority level distribution
    print("\nSeniority level distribution:")
    print(cluster_data['Seniority level'].value_counts(normalize=True) * 100)
    
    # Country distribution
    print("\nCountry distribution:")
    print(cluster_data['Country'].value_counts(normalize=True) * 100)
    
    # Average annual salary - Now correctly calculated since 'Annual salary' is numeric
    print(f"\nAverage Annual Salary: {cluster_data['Annual salary'].mean():,.2f}")


# Initialize a list to hold summary data for each cluster and save analysis to csv
cluster_summaries = []


cluster_analysis = df.groupby('Cluster')[rank_columns].mean()
cluster_sizes = df.groupby('Cluster').size()

cluster_sizes_df = cluster_sizes.to_frame(name='Cluster Size').reset_index()
summary_df = cluster_analysis.merge(cluster_sizes_df, on='Cluster')
summary_df.to_csv('cluster_summary_with_sizes.csv', index=False)

details_df = df[['Cluster', 'Age', 'Annual salary', 'Country']]
details_df = details_df.sort_values(by='Cluster')
details_df.to_csv('cluster_details.csv', index=False)

