# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:11:00 2024

@author: Fik

# https://geo.rae.gr/?tab=panel-1339

"""

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


def plot_geo_csv(geo_csv):
    geo_csv.plot()
    plt.show()
    plt.clf()
    plt.close()


# Function to process each CSV file
def process_csv_file(csv_file_path):
    # Load the CSV file with semicolon delimiter and specify dtype to avoid mixed types warning
    df = pd.read_csv(csv_file_path, delimiter=';', header=None, dtype=str)
    
    # Rename the columns for better understanding
    df.columns = ['Longitude', 'Latitude', 'Wind_Speed']
    
    # Convert the Longitude and Latitude columns to numeric types
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    
    # Drop rows with invalid values
    df = df.dropna(subset=['Longitude', 'Latitude'])
    
    # Create a GeoDataFrame
    geometry = gpd.points_from_xy(df['Longitude'], df['Latitude'])
    gdf_csv = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:2100")

    # Plot the GeoDataFrame
    #plot_geo_csv(gdf_csv)

    
    # Re-project the coordinates to a geographic CRS (EPSG:4326)
    gdf_csv = gdf_csv.to_crs(epsg=4326)
    
    # Extract the transformed coordinates
    df['Longitude'] = gdf_csv.geometry.x
    df['Latitude'] = gdf_csv.geometry.y
    
    # Plot the reprojected GeoDataFrame
    #plot_geo_csv(gdf_csv)
    
    return df


def main(): 
    pd.set_option("display.max.columns", None)
    
    csv_directories_list = ["G:/DWS/IWW-02-07/project 02/wind_speed_csvs/h80",
                            "G:/DWS/IWW-02-07/project 02/wind_speed_csvs/h100",
                            "G:/DWS/IWW-02-07/project 02/wind_speed_csvs/h120"]
    for csv_directory in csv_directories_list:

        # Iterate through the CSV files in the directory
        for file in os.listdir(csv_directory):
            if file.endswith(".csv"):
                csv_file_path = os.path.join(csv_directory, file)
                df_transformed = process_csv_file(csv_file_path)
                
                # Create the output file path with "_4326" suffix
                output_file_path = os.path.join(csv_directory, file.replace('.csv', '_4326.csv'))
                
                # Save the transformed DataFrame back to a CSV file
                df_transformed.to_csv(output_file_path, index=False, sep=';')
                
                print(f"Processed and saved {output_file_path}")


if __name__ == '__main__':
    main()