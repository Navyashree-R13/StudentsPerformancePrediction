import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('desktop/DAPM_Excel/overall_dis_encoded.csv')  # Replace with the path to your dataset
scaler = MinMaxScaler()
df['overall score'] = scaler.fit_transform(df[['overall score']])
df.to_csv('desktop/DAPM_Excel/ode_normalization.csv', index=False)  # Replace with your desired save path
