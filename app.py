# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:40:31 2024

@author: kasra
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Helper functions
def apply_pca(data):
    """Apply PCA to standardized data and return PCA data and explained variance."""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    pca = PCA()
    pca_data = pca.fit_transform(standardized_data)
    explained_variance = pca.explained_variance_ratio_
    return pca_data, explained_variance

def determine_components_for_variance(explained_variance, threshold=0.8):
    """Determine the number of components needed to reach a variance threshold."""
    cumulative_variance = np.cumsum(explained_variance)
    num_components = np.where(cumulative_variance >= threshold)[0][0] + 1
    return num_components, cumulative_variance

def create_social_vulnerability_index(pca_data, num_components):
    """Create a Social Vulnerability Index (SoVI) using PCA components."""
    sovi = pca_data[:, :num_components]
    sovi_scores = np.mean(sovi, axis=1)
    return sovi_scores

def scale_sovi(sovi_scores):
    """Scale SoVI scores to a 0-100 range."""
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_scores = scaler.fit_transform(sovi_scores.reshape(-1, 1)).flatten()
    return scaled_scores

def preprocess_data(df, variable_columns):
    """Preprocess data by handling missing values and normalizing columns."""
    # Report and remove rows with NaN values
    nan_rows = df[df[variable_columns].isna().any(axis=1)]
    if not nan_rows.empty:
        st.write("### Rows with Missing Values Removed")
        for idx, row in nan_rows.iterrows():
            missing_columns = row[row.isna()].index.tolist()
            st.write(f"Row {idx} removed due to missing value(s) in column(s): {missing_columns}")
        df = df.dropna(subset=variable_columns)
    
    # Normalize variables (scaling each column to make the max value 1)
    df[variable_columns] = df[variable_columns].apply(lambda x: x / x.max() if x.max() != 0 else 0, axis=0)
    
    return df

# Streamlit app starts here
st.title("Social Vulnerability Index Calculator")

st.write("""
Upload your properly formatted Excel file to calculate the Social Vulnerability Index (SoVI) using PCA.
The app will automatically detect variables, preprocess data, generate visualizations, and allow you to download results.
""")

# File upload
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])
if uploaded_file is not None:
    # Load the data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Data Preview")
    st.write(df.head())

    # Automatically identify variable columns (all columns except the first one)
    variable_columns = df.columns[1:]
    st.write(f"Identified variables for PCA: {list(variable_columns)}")

    try:
        # Preprocess data
        df = preprocess_data(df, variable_columns)
        
        # Run PCA analysis
        pca_data, explained_variance = apply_pca(df[variable_columns])
        num_components, cumulative_variance = determine_components_for_variance(explained_variance)
        df['SoVI'] = create_social_vulnerability_index(pca_data, num_components)
        df['SoVI_Scaled'] = scale_sovi(df['SoVI'].values)
        
        # Display results
        st.write("### Results")
        st.write(f"Number of components to explain 80% variance: {num_components}")
        st.write(df[['SoVI', 'SoVI_Scaled']].head())
        
        # Download link
        csv = df.to_csv(index=False)
        st.download_button("Download Results", csv, "results.csv", "text/csv")

        # Plot correlation with SoVI
        st.write("### Correlation with SoVI")
        standardized_df = pd.DataFrame(StandardScaler().fit_transform(df[variable_columns]), columns=variable_columns)
        standardized_df['SoVI'] = df['SoVI_Scaled'].values
        corr_matrix = standardized_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix[['SoVI']].sort_values(by='SoVI', ascending=False), 
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)

        # Plot bivariate correlations for variable columns
        st.write("### Bivariate Correlations Among Variables")
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(standardized_df[variable_columns].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing the file: {e}")
