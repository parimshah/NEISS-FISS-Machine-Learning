import pandas as pd

# 1993-2015 NEISS-FISS Data
df = pd.read_stata("C://Users//rohin//Documents//gen//Weil Cornell//NEISS Data//37276-0001-Data.dta")

# Removing BB gun injuries & non-gunshot wound injuries
bb_columns = ["FA_NGSW", "BB_GSW", "BB_NGSW"]
for x in bb_columns:
    df = df[df[x] != 'Yes']
    df = df.dropna(subset=[x])

df.reset_index(drop=True, inplace=True)


# Converting intent values to numbers
value_mapping = {
    'Unknown': 0,
    'Unintentnl': 1,
    'Assault': 2,
    'Suicide': 3,
    'Law enforce': 4
}
df['CLASS_C'] = df['CLASS_C'].replace(value_mapping)

df.to_csv('intentNEISSData.csv')



