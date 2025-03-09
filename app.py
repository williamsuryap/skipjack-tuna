
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Skipjack Tuna Analysis Dashboard",
    # layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('skipjack_data.csv', encoding='UTF-8-SIG')
    
    # Create a Season column
    def assign_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(assign_season)
    
    # Create a Year-Month column for time series
    df['YearMonth'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')
    
    return df

# Load the data
df = load_data()

# Sidebar
st.sidebar.title("Skipjack Tuna Analysis")
st.sidebar.image("https://fian-indonesia.org/wp-content/uploads/2021/03/skipjack-tuna-768x768.jpg", width=200)

# Navigation
page = st.sidebar.radio("Select Page", ["Overview", "Seasonal Analysis", "Environmental Factors", "Spatial Analysis"])

# Overview Page
if page == "Overview":
    st.title("Skipjack Tuna Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"Total Records: {len(df)}")
        st.write(f"Date Range: {df['Year'].min()} to {df['Year'].max()}")
        st.write(f"Average Catch: {df['Catch (kg)'].mean():.2f} kg")
        st.write(f"Maximum Catch: {df['Catch (kg)'].max()} kg")
    
    with col2:
        st.subheader("Data Sample")
        st.dataframe(df.head())
    
    st.subheader("Catch Distribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(df['Catch (kg)'].clip(upper=5000), bins=50, kde=True, color='#766CDB', ax=ax)
    ax.set_title('Distribution of Catch Weights', fontsize=20, fontweight='semibold', color='#222222', pad=15)
    ax.set_xlabel('Catch (kg)', fontsize=16, color='#333333', labelpad=10)
    ax.set_ylabel('Frequency', fontsize=16, color='#333333', labelpad=10)
    ax.tick_params(labelsize=14, colors='#555555')
    ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    plt.tight_layout()
    st.pyplot(fig)

# Seasonal Analysis Page
elif page == "Seasonal Analysis":
    st.title("Seasonal Patterns in Skipjack Tuna Catch")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Catch by Season")
        seasonal_avg = df.groupby('Season')['Catch (kg)'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(x='Season', y='Catch (kg)', data=seasonal_avg, 
                   order=['Winter','Spring','Summer','Fall'], 
                   palette=['#766CDB','#DA847C','#D9CC8B','#7CD9A5'], ax=ax)
        ax.set_title('Average Catch by Season', fontsize=20, fontweight='semibold', color='#222222', pad=15)
        ax.set_xlabel('Season', fontsize=16, color='#333333', labelpad=10)
        ax.set_ylabel('Average Catch (kg)', fontsize=16, color='#333333', labelpad=10)
        ax.tick_params(labelsize=14, colors='#555555')
        ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Monthly Trends")
        monthly_avg = df.groupby('Month')['Catch (kg)'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.lineplot(x='Month', y='Catch (kg)', data=monthly_avg, marker='o', color='#766CDB', ax=ax)
        ax.set_title('Average Catch by Month', fontsize=20, fontweight='semibold', color='#222222', pad=15)
        ax.set_xlabel('Month', fontsize=16, color='#333333', labelpad=10)
        ax.set_ylabel('Average Catch (kg)', fontsize=16, color='#333333', labelpad=10)
        ax.set_xticks(range(1, 13))
        ax.tick_params(labelsize=14, colors='#555555')
        ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
        st.pyplot(fig)
    
    st.subheader("Seasonal Catch Distribution")
    
    selected_season = st.selectbox("Select Season", ['All Seasons', 'Winter', 'Spring', 'Summer', 'Fall'])
    
    if selected_season == 'All Seasons':
        filtered_df = df
    else:
        filtered_df = df[df['Season'] == selected_season]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(x='Season', y='Catch (kg)', data=filtered_df if selected_season == 'All Seasons' else df[df['Season'] == selected_season], 
               palette=['#766CDB','#DA847C','#D9CC8B','#7CD9A5'], ax=ax)
    ax.set_title(f'Catch Distribution by Season ({selected_season})', fontsize=20, fontweight='semibold', color='#222222', pad=15)
    ax.set_xlabel('Season', fontsize=16, color='#333333', labelpad=10)
    ax.set_ylabel('Catch (kg)', fontsize=16, color='#333333', labelpad=10)
    ax.tick_params(labelsize=14, colors='#555555')
    ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    st.pyplot(fig)

# Environmental Factors Page
elif page == "Environmental Factors":
    st.title("Environmental Factors Affecting Skipjack Tuna Catch")
    
    # Select environmental factor
    env_factor = st.selectbox("Select Environmental Factor", 
                             ["Sea Surface Temperature (SST)", 
                              "Chlorophyll (CHL)", 
                              "Mixed Layer Depth (MLD)", 
                              "Salinity (Salin)"])
    
    factor_map = {
        "Sea Surface Temperature (SST)": "SST",
        "Chlorophyll (CHL)": "CHL",
        "Mixed Layer Depth (MLD)": "MLD",
        "Salinity (Salin)": "Salin"
    }
    
    selected_factor = factor_map[env_factor]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Catch vs {env_factor}")
        
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(x=selected_factor, y='Catch (kg)', data=df, alpha=0.5, color='#766CDB', ax=ax)
        
        ax.set_title(f'Catch vs {env_factor}', fontsize=20, fontweight='semibold', color='#222222', pad=15)
        ax.set_xlabel(env_factor, fontsize=16, color='#333333', labelpad=10)
        ax.set_ylabel('Catch (kg)', fontsize=16, color='#333333', labelpad=10)
        ax.tick_params(labelsize=14, colors='#555555')
        ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
        st.pyplot(fig)
    
    with col2:
        st.subheader(f"{env_factor} by Season")
        
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(x='Season', y=selected_factor, data=df, 
                   order=['Winter','Spring','Summer','Fall'], 
                   palette=['#766CDB','#DA847C','#D9CC8B','#7CD9A5'], ax=ax)
        ax.set_title(f'{env_factor} by Season', fontsize=20, fontweight='semibold', color='#222222', pad=15)
        ax.set_xlabel('Season', fontsize=16, color='#333333', labelpad=10)
        ax.set_ylabel(env_factor, fontsize=16, color='#333333', labelpad=10)
        ax.tick_params(labelsize=14, colors='#555555')
        ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
        st.pyplot(fig)
    
    st.subheader("Correlation Matrix")
    
    # Calculate correlation matrix
    corr_matrix = df[['Catch (kg)', 'SST', 'CHL', 'MLD', 'Salin']].corr()
    
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Environmental Factors', fontsize=20, fontweight='semibold', color='#222222', pad=15)
    ax.tick_params(labelsize=14, colors='#555555')
    st.pyplot(fig)

# Spatial Analysis Page
elif page == "Spatial Analysis":
    st.title("Spatial Distribution of Skipjack Tuna Catch")
    
    st.write("This page shows the spatial distribution of skipjack tuna catch.")
    
    # Filter by season
    selected_season = st.selectbox("Filter by Season", ['All Seasons', 'Winter', 'Spring', 'Summer', 'Fall'])
    
    if selected_season == 'All Seasons':
        filtered_df = df
    else:
        filtered_df = df[df['Season'] == selected_season]
    
    # Create scatter plot on map
    st.subheader(f"Catch Distribution Map ({selected_season})")
    
    # Create the map using Plotly Express
    fig = px.scatter_mapbox(
        filtered_df,
        lat="Lat",
        lon="Lon",
        color="Catch (kg)",
        size="Catch (kg)",
        hover_data=['SST', 'CHL', 'MLD', 'Salin', 'Date'],
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=15,
        zoom=5,
        height=600,
        width=800
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig.update_layout(
        title=f'Spatial Distribution of Skipjack Tuna Catch ({selected_season})',
        title_font=dict(size=20)
    )
    
    st.plotly_chart(fig)
