# Install dependencies as needed:
# !pip3 install -U ucimlrepo 

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
# data (as pandas dataframes) 
diabetes_130_features = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
diabetes_130_targets = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 
  
# metadata 
print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 
  
# variable information 
print(diabetes_130_us_hospitals_for_years_1999_2008.variables) 


diabetes_130_features.to_csv("diabetes_130_features.csv", index=False)
diabetes_130_targets.to_csv("diabetes_130_targets.csv", index=False)