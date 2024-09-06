import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample Data (from the extended dataset)
extended_data = pd.DataFrame({
    'Policyholder Age': np.random.choice([25, 30, 35, 40, 45, 48, 50, 55, 60], size=50),
    'Vehicle Age': np.random.choice([3, 5, 6, 8, 10, 12, 14, 15, 18, 20], size=50),
    'Claim Amount': np.random.choice([1500, 2000, 2500, 3000, 7000, 8000, 9000, 10000, 11000, 13000], size=50),
    'Number of Claims': np.random.choice([0, 1, 2, 3, 4], size=50),
    'Premium Paid': np.random.choice([400, 450, 500, 600, 900, 950, 1000, 1100, 1200, 1500], size=50)
})

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(extended_data)

# User input to select the number of principal components
n_components = st.sidebar.radio('Select number of principal components', [2, 3])

# Apply PCA
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame for the PCA result
pca_columns = [f'PC_{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(pca_result, columns=pca_columns)

# Explained variance
explained_variance = pca.explained_variance_ratio_

# Streamlit app for PCA visualization
st.title('PCA with 2D or 3D Visualization')

# Display original data
st.write('### Original Dataset')
st.write(extended_data.head())

# Display PCA-transformed data
st.write(f'### PCA Transformed Data (Using {n_components} Components)')
st.write(pca_df.head())

# 2D Visualization if 2 components are selected
if n_components == 2:
    st.write("### 2D PCA Plot")
    fig, ax = plt.subplots()
    ax.scatter(pca_df['PC_1'], pca_df['PC_2'], color='cyan')
    ax.set_xlabel('Principal Component 1', color='magenta')
    ax.set_ylabel('Principal Component 2', color='magenta')
    ax.set_title('PCA 2D Visualization (2 Components)', color='yellow')
    ax.tick_params(axis='x', colors='cyan')
    ax.tick_params(axis='y', colors='cyan')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    st.pyplot(fig)

# 3D Visualization if 3 components are selected
elif n_components == 3:
    st.write("### 3D PCA Plot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot for PCA 3D components
    ax.scatter(pca_df['PC_1'], pca_df['PC_2'], pca_df['PC_3'], color='cyan')
    
    # Set axis labels
    ax.set_xlabel('Principal Component 1', color='magenta')
    ax.set_ylabel('Principal Component 2', color='magenta')
    ax.set_zlabel('Principal Component 3', color='magenta')
    ax.set_title('PCA 3D Visualization (3 Components)', color='yellow')
    
    # Set face color for 3D plot
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    st.pyplot(fig)

