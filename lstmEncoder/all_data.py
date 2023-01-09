import glob
import pandas as pd

all_data = []
for i in glob.glob("..\ext*.csv"):
    print(i)
    temp = pd.read_csv(i)
    temp['label'] = i.split('-')[-1].split('.')[0]
    temp['combined'] = temp['workLabel'] + " is a " + temp['workDesc']
    all_data.append(temp[['combined', 'label']])
pd.concat(all_data, axis=0).to_csv('combined.tsv', sep='\t', index=False)