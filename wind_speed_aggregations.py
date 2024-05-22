# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:36:01 2024

@author: Fik

https://gadm.org/download_country.html

"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# Function to print CRS and plot the shapefile
def process_shapefile(shapefile_path):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Print the CRS
    print(f"CRS of {os.path.basename(shapefile_path)}: {gdf.crs}")
    
    # Plot the shapefile
    gdf.plot()
    plt.title(f"Plot of {os.path.basename(shapefile_path)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
    # Print the first few rows to check for name attribute
    print(gdf.head())
    
    # Try to print the names of regional units if a relevant column is found
    name_column = None
    for col in gdf.columns:
        if 'name' in col.lower() or 'region' in col.lower():
            name_column = col
            break
    
    if name_column:
        print(f"Names of regional units in {os.path.basename(shapefile_path)}:")
        print(gdf[name_column])
    else:
        print(f"No name or region column found in {os.path.basename(shapefile_path)}")


def shape_files_details(shapefile_directory):
    for file in os.listdir(shapefile_directory):
        if file.endswith(".shp"):
            shapefile_path = os.path.join(shapefile_directory, file)
            process_shapefile(shapefile_path)   


def calculate_groupBy_avgs(csv_path, gdf_map, groupby_list_of_cols, calculation_col, char, aggr_mean_df):
    df = pd.read_csv(csv_path, delimiter=';')
    df = df.rename(columns=lambda x: x.lower())
    df = df[~(df == 0).any(axis=1)]

    # Convert CSV data to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    
    # Spatial join to map points to municipalities and districts
    gdf = gpd.sjoin(gdf, gdf_map, how="left", op="within")

    # Calculate mean wind speed for each municipality and district
    mean_df = gdf.groupby(groupby_list_of_cols, as_index=False)[calculation_col].mean()
    
    # Rename wind_speed column to wind_speed_avg_{char}
    mean_df = mean_df.rename(columns={calculation_col: f'{calculation_col}_avg_{char}'})
    #print(f"mean_df for {char}:");print(mean_df.head())
    
    # Merge with the aggregated DataFrame
    if aggr_mean_df.empty:
        aggr_mean_df = mean_df
    else:
        aggr_mean_df = pd.merge(aggr_mean_df, mean_df, on=groupby_list_of_cols, how='outer')

    return aggr_mean_df



def main(): 
    pd.set_option("display.max.columns", None)
    
    shapefile_directory = "G:/DWS/IWW-02-07/project 02/gadm41_GRC_shp"
    
    csv_directories = ["G:/DWS/IWW-02-07/project 02/wind_speed_csvs/h80/",
                       "G:/DWS/IWW-02-07/project 02/wind_speed_csvs/h100/",
                       "G:/DWS/IWW-02-07/project 02/wind_speed_csvs/h120/"]
    
    # Iterate through the shapefiles in the directory
    #shape_files_details(shapefile_directory)
    
    '''
    After calling shape_files_details we end up considering 
    interesting the gadm41_GRC_3.shp file
    '''
    
    gdf_map= gpd.read_file( os.path.join(shapefile_directory, 'gadm41_GRC_3.shp'))  
    
    # Select and rename the columns
    gdf_map = gdf_map[['NAME_3', 'NL_NAME_3', 'NAME_2', 'geometry']]
    gdf_map = gdf_map.rename(columns={'NAME_3': 'munic_eng','NL_NAME_3': 'munic_greek',
                             'NAME_2': 'district'})
    #print(type(gdf_map), gdf_map.head())
    
    
    # Initialize an empty DataFrame to store aggregated results
    calculation_col = 'wind_speed'
    
    groupby_list_of_cols_munic = ['munic_eng', 'munic_greek', 'district']
    aggr_mean_munic = pd.DataFrame(columns=['munic_eng', 'munic_greek', 'district',
                                      'wind_speed_avg_h80', 'wind_speed_avg_h100', 'wind_speed_avg_h120'])
    
    groupby_list_of_cols_distr = ['district']
    aggr_mean_distr = pd.DataFrame(columns=['district','wind_speed_avg_h80', 'wind_speed_avg_h100', 'wind_speed_avg_h120'])
    
    for csv_directory in csv_directories:
        char = csv_directory.split('/')[-2] 
        # Iterate through each CSV file in the directory
        csv_path = os.path.join(csv_directory, f"00_Wind_Speed_CSV_Greece_({char})_4326.csv")      

        aggr_mean_munic = calculate_groupBy_avgs(csv_path, gdf_map, groupby_list_of_cols_munic, calculation_col, char, aggr_mean_munic)
        print(aggr_mean_munic)

        aggr_mean_distr = calculate_groupBy_avgs(csv_path, gdf_map, groupby_list_of_cols_distr, calculation_col, char, aggr_mean_distr)
        print(aggr_mean_distr)

    # Save the aggregated results to a CSV file
    aggr_mean_munic.to_csv("aggregated_wind_speed_munic.csv", index=False)
    aggr_mean_distr.to_csv("aggregated_wind_speed_distr.csv", index=False) 
    
    

if __name__ == '__main__':
    main()