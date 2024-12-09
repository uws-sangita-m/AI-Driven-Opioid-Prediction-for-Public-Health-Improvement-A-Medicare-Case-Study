#Import Packages
import os
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Set environment variable 
os.environ['OMP_NUM_THREADS'] = '4'

# Load the Opioid Cliams and Providers dataset from the CMS API
Opioid_claims_data_url = "https://data.cms.gov/data-api/v1/dataset/c37ebe6d-f54f-4d7d-861f-fefe345554e6/data"
providers_data_url = "https://data.cms.gov/data-api/v1/dataset/f1a8c197-b53d-4c24-9770-aea5d5a97dfb/data"

# Read API requests
response = requests.get(Opioid_claims_data_url)
providers_response = requests.get(providers_data_url)

if providers_response.status_code == 200:
    
    # Load and summarize provider data
    providers_data = pd.DataFrame(providers_response.json())
    provider_counts = providers_data.groupby('STATE').size().reset_index(name='Num_Providers')
    provider_counts = provider_counts.sort_values(by='Num_Providers', ascending=False)


    # Plot number of providers per state
    plt.figure(figsize=(12, 6))
    sns.barplot(data=provider_counts, x='STATE', y='Num_Providers')
    plt.xticks(rotation=90)
    plt.title('Distribution of Opioid Treatment Providers by State',fontsize = 20)
    plt.xlabel('State')
    plt.ylabel('Number of Providers')
    plt.show()

else:
    print(f"Failed to fetch provider data. Status code: {providers_response.status_code}")


# Check for successful response and load data
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)

    # Clean and preprocess the data
    #df = df.dropna()  # Drop missing values for simplicity

    # Convert relevant columns to numeric where possible
    numeric_cols = ['Year', 'Tot_Clms', 'Opioid_Prscrbng_Rate', 'LA_Tot_Opioid_Clms', 'Tot_Opioid_Clms']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Balancing Data - Last Observation Carried Forward (LOCF) 
    df.fillna(method='ffill', inplace=True)
    time_df = df   
    c4_df = df
    c4_df = c4_df[(c4_df['Geo_Lvl'] != 'National')]   
    fil_time_df = time_df[(time_df['Geo_Lvl'] == 'National')]
    time_df_group = fil_time_df.groupby('Year').agg(LA_Tot_Opioid_Clms=('LA_Tot_Opioid_Clms', 'sum')).reset_index()

    
    # Identify high-risk regions by grouping opioid claims by region
    high_risk_regions = df.groupby('Geo_Desc')['Tot_Opioid_Clms'].sum().reset_index()
    high_risk_regions = high_risk_regions.sort_values(by='Tot_Opioid_Clms', ascending=False)
    high_risk_regions_top_10 = high_risk_regions[high_risk_regions['Geo_Desc'] != 'National']   
    tst = high_risk_regions.head(11)['Geo_Desc']
    filtered_tst = tst[tst != 'National']

    first_five = filtered_tst.head(6).reset_index(drop=True)
    to_list = first_five.tolist()
    
    #Model Preparation
    for i in to_list:
        fil_c4_df_1 = c4_df[(c4_df['Geo_Desc'] == i)]
        fil_c4_df_1.to_csv('cp_model.csv', index=False)
        features_1 = fil_c4_df_1[['Year', 'Tot_Clms', 'Opioid_Prscrbng_Rate']]
        target_1 = fil_c4_df_1['Tot_Opioid_Clms']

        # Split the data into training and testing sets
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(features_1, target_1, test_size=0.3, random_state=42)

        # Train a Random Forest model
        rf_model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model_1.fit(X_train_1, y_train_1)

        # Evaluate the Random Forest model
        y_pred_rf_1 = rf_model_1.predict(X_test_1)
        rfr2_1=r2_score(y_test_1, y_pred_rf_1)
        rf_mse_1 = mean_squared_error(y_test_1, y_pred_rf_1)
        print(i,"Random Forest R2 Score:", r2_score(y_test_1, y_pred_rf_1))
        print(i,"Random Forest Mean Squared Error:", mean_squared_error(y_test_1, y_pred_rf_1))
        
        # Train a Decision Tree model 
        dt_model_1 = DecisionTreeRegressor(random_state=42)
        dt_model_1.fit(X_train_1, y_train_1)

        # Evaluate the Decision Tree model
        y_pred_dt_1 = dt_model_1.predict(X_test_1)
        accuracy_dt_1 = r2_score(y_test_1, y_pred_dt_1)
        mse_dt_1 = mean_squared_error(y_test_1, y_pred_dt_1)
        mae_dt_1 = mean_absolute_error(y_test_1, y_pred_dt_1)
        precision_dt_1 = 1 - (mae_dt_1 / y_test_1.mean())
        dtr2_1= r2_score(y_test_1, y_pred_dt_1)
        dt_mse_1 = mean_squared_error(y_test_1, y_pred_dt_1)    
        print(i,"Decision Tree R2 Score:", r2_score(y_test_1, y_pred_dt_1))
        print(i,"Decision Tree Mean Squared Error:", mean_squared_error(y_test_1, y_pred_dt_1))
        #print(mse_dt_1,"   ",mae_dt_1)
        
        # Train a Support Vector Model(SVF)
        sv_model_1 = SVR(kernel='rbf')
        sv_model_1.fit(X_train_1, y_train_1)
        
        # Evaluate the SVF model
        y_pred_sv_1 = sv_model_1.predict(X_test_1)
        svr2_1 = r2_score(y_test_1, y_pred_sv_1)
        sv_mse_1 = mean_squared_error(y_test_1, y_pred_sv_1)
        print(i,"SV R2 Score:", r2_score(y_test_1, y_pred_sv_1))
        print(i,"SV Mean Squared Error:", mean_squared_error(y_test_1, y_pred_sv_1))
        
    # Plot high-risk regions
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Tot_Opioid_Clms', y='Geo_Desc', data=high_risk_regions_top_10.head(10))
    plt.title('Top 10 High-Risk States with Highest Opioid Claims',fontsize=20)
    plt.xlabel('Total Opioid Claims',fontsize=15)
    plt.ylabel('Region',fontsize=15)
    plt.show()

    # Prepare data for machine learning
    features = df[['Year', 'Tot_Clms', 'Opioid_Prscrbng_Rate']]
    target = df['Tot_Opioid_Clms']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the Random Forest model
    y_pred_rf = rf_model.predict(X_test)
    rfr2=r2_score(y_test, y_pred_rf)
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
    print("Random Forest Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))


    # Train a Decision Tree model for comparison
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)

    # Evaluate the Decision Tree model
    y_pred_dt = dt_model.predict(X_test)
    accuracy_dt = r2_score(y_test, y_pred_dt)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    precision_dt = 1 - (mae_dt / y_test.mean())
    dtr2= r2_score(y_test, y_pred_dt)
    dt_mse = mean_squared_error(y_test, y_pred_dt)
    print("Decision Tree R2 Score:", r2_score(y_test, y_pred_dt))
    print("Decision Tree Mean Squared Error:", mean_squared_error(y_test, y_pred_dt))
    print(mse_dt,"   ",mae_dt)

    # Calculate residuals for Decision Tree
    residuals_dt = y_test - y_pred_dt
    
    # Plotting the residual plot for Decision Tree Regressor
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred_dt, y=residuals_dt, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residual Plot for Decision Tree Regressor')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()
    

    # Use K-Means clustering to identify unusual patterns
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)   
    sv_model = SVR(kernel='rbf')
    sv_model.fit(X_train, y_train)
    
    # Evaluate the SVF model
    y_pred_sv = sv_model.predict(X_test)
    svr2 = r2_score(y_test, y_pred_sv)
    sv_mse = mean_squared_error(y_test, y_pred_sv)
    print("SV R2 Score:", r2_score(y_test, y_pred_sv))
    print("SV Mean Squared Error:", mean_squared_error(y_test, y_pred_sv))  
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    models = ['Random Forest', 'Decision Tree', 'Support Vector']
    r2_values = [rfr2, dtr2, svr2]
    
    # Plotting the R2 Score comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models, r2_values, color=['blue', 'orange', 'red'])
    plt.title('R2 Score Comparison')
    plt.ylabel('R2 Score')
    plt.xlabel('Model')
    plt.show()

    # Provided MSE scores
    models = ['Random Forest', 'Decision Tree', 'Support Vector']
    mse_values = [rf_mse, dt_mse, sv_mse]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # R2 Score comparison
    axes[0].bar(models, r2_values, color=['blue', 'orange', 'red'])
    axes[0].set_title('R2 Score Comparison',fontsize=15)
    axes[0].set_ylabel('R2 Score',fontsize=15)
    axes[0].set_xlabel('Model',fontsize=15)

    # MSE comparison
    axes[1].bar(models, mse_values, color=['blue', 'orange', 'red'])
    axes[1].set_title('Mean Square Error (MSE) Comparison',fontsize=15)
    axes[1].set_ylabel('MSE Score',fontsize=15)
    axes[1].set_xlabel('Model',fontsize=15)
    fig.suptitle("Model Comparision Metrics", fontsize=25)
    plt.tight_layout()
    plt.show()
    # Plotting the R2 Score comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models, mse_values, color=['blue', 'orange', 'red'])
    plt.title('Mean Square Error Comparison')
    plt.ylabel('MSE Score')
    plt.xlabel('Model')
    plt.show()
    
    # Plot the clustering results
    plt.figure(figsize=(10, 6))
    #sns.scatterplot(data=df, x='Opioid_Prscrbng_Rate', y='LA_Tot_Opioid_Clms', hue='Cluster', palette='deep')
    sns.scatterplot(data=df, x='Tot_Clms', y='Tot_Opioid_Clms', hue='Cluster', palette='deep')
    plt.title('Clustering of Opioid Prescribing Patterns')
    plt.show()

else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

data = pd.read_csv("Total_Claims.csv")
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
recent_years = data[data['Year'] >= 2013] 
recent_years['Tot_Opioid_Clms'] = pd.to_numeric(recent_years['Tot_Opioid_Clms'], errors='coerce')
recent_years = recent_years[recent_years['Prscrbr_Geo_Lvl']== 'County']
recent_years = recent_years[recent_years['Breakout_Type']== 'Totals']
recent_years[['split_State', 'split_County']] = recent_years['Prscrbr_Geo_Desc'].str.split(':', expand=True)
state_list = ['California', 'Texas', 'Florida', 'New York', 'Ohio'] 
state_list = to_list
#print(state_list)
fil_df = recent_years[recent_years['split_State'].isin(state_list)]
# Group by Year and State, then sum the opioid claims
state_claims = fil_df.groupby(['split_County', 'split_State'])['Tot_Opioid_Clms'].sum().reset_index()
#print(state_claims)

# Create a 2x3 grid for pie charts
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#fig, axes = plt.subplots(2, 3, figsize=(17, 11))
axes = axes.ravel()  # Flatten the axes array for easy iteration

for i,state in enumerate(state_list):
    county_data = state_claims[state_claims['split_State'] == state].sort_values(by='Tot_Opioid_Clms', ascending=False).head(6)
    colors = ['red', 'orange','yellow', 'violet', 'pink','green']
    if len(county_data) < len(colors):
        colors = colors[:len(county_data)]
    axes[i].pie(county_data['Tot_Opioid_Clms'], labels=county_data['split_County'], autopct='%1.1f%%', startangle=140,colors=colors)
    axes[i].set_title(f"{state}", fontsize=20)
    
# Normalize the values to determine the color scale
max_claims = county_data['Tot_Opioid_Clms'].max()
normalized_values = county_data['Tot_Opioid_Clms'] / max_claims


colors = [
    (1, value, 0) for value in 1 - normalized_values
] 


# Add a main title to the entire grid
fig.suptitle("Top Six Counties with Highest Opioid Claims in the Leading Six US States", fontsize=25)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()
output_file_path = 'fil_testing.csv'  # Specify your desired file path
county_data.to_csv(output_file_path, index=False)
df = df[df['Geo_Desc'] != 'National']
df = df[df['Geo_Lvl'] == 'State']
data = df[['Tot_Clms', 'Tot_Opioid_Clms','Geo_Desc']].dropna()  # Ensure no missing values
data.to_csv(output_file_path, index=False)
data_grouped = data.groupby('Geo_Desc').agg({'Tot_Opioid_Clms': 'sum', 'Tot_Clms': 'sum'}).reset_index()
data = data_grouped
data.to_csv(output_file_path, index=False)
kmeans = KMeans(n_clusters=3, random_state=42)  # Example: 3 clusters
data['Cluster'] = kmeans.fit_predict(data[['Tot_Clms', 'Tot_Opioid_Clms']])


# Visualize the clustering
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data['Tot_Clms'], data['Tot_Opioid_Clms'], c=data['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Total Claims')
plt.ylabel('Total Opioid Claims')
plt.title('Clustering of Total Claims vs. Total Opioid Claims')

for i, row in data.iterrows():
    plt.text(row['Tot_Clms'], row['Tot_Opioid_Clms'], row['Geo_Desc'], fontsize=8, ha='right', va='bottom')

# Add color legend for clusters
plt.colorbar(scatter, label='Cluster')
plt.show()


# Setting up the data for ARIMA model
time_df_group.set_index('Year', inplace=True)
# Building and fitting the ARIMA model
model = ARIMA(time_df_group, order=(2,1,2))  # ARIMA(p, d, q) where p, d, q are model parameters
model_fit = model.fit()

# Making predictions for the next 10 years
future_years = [time_df_group.index[-1] + i for i in range(1, 11)]
forecast = model_fit.get_forecast(steps=10)
forecast_values = forecast.predicted_mean

# Plotting the historical data and predictions
plt.figure(figsize=(12, 6))
plt.plot(time_df_group.index, time_df_group['LA_Tot_Opioid_Clms'], label='Historical Total Opioid Claims')
plt.plot(future_years, forecast_values, marker='o', linestyle='--', label='Predicted Total Opioid Claims (ARIMA)')

# Adding titles and labels
plt.title('ARIMA Time Series Prediction for Total Opioid Claims (Next 10 Years)')
plt.xlabel('Year')
plt.ylabel('Total Opioid Claims')
plt.xticks(list(time_df_group.index) + future_years, rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Displaying the predicted values for the next 10 years
predicted_df = pd.DataFrame({
    'Year': future_years,
    'Predicted_Total_Opioid_Clms': forecast_values
})
print(predicted_df)




