import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


code_df = pd.read_csv(os.path.join('..', '..','data', 'raw', 'country_codes.csv'))
code_df = code_df[['Name', 'alpha-3']]


# some countries got lost in this merge, can come back and edit the names if desired, but this will be replace with full international model anyway
merged_df = pd.merge(country_df, code_df, left_on='Dep IATA Country Name', right_on='Name')


population_df = pd.read_csv(os.path.join('flight_data', 'country_population.csv'))[['Country Code', '2021']]

final_df = pd.merge(merged_df, population_df, left_on='alpha-3', right_on='Country Code')[['Name','Seats (Total)', 'Country Code', '2021']]
final_df.rename(columns={'Seats (Total)':'Seats', '2021':'Population'}, inplace=True)

final_df.to_csv(os.path.join('..', '..','data', 'processed', 'country_flights.csv'))