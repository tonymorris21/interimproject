import pandas as pd
import pyarrow.feather as feather

df = pd.read_csv('car_evaluation.csv')
df.to_feather('test.feather')
df = pd.read_feather('C:\\Users\\Tony\\Desktop\\interimproject\\uploads\\test.feather')
print(df.head)