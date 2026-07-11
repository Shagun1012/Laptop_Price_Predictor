import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('laptop_data.csv')

# Basic checks
df.drop(columns=['Unnamed: 0'], inplace=True)

# Clean columns
df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

#  EDA 
sns.displot(df['Price'])

sns.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.barplot(x=df['TypeName'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.scatterplot(x=df['Inches'], y=df['Price'])
plt.show()

#  Feature Engineering

# Touchscreen
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

# IPS
df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

# Resolution
new = df['ScreenResolution'].str.split('x', n=1, expand=True)

df['X_res'] = new[0]
df['Y_res'] = new[1]

# FIXED extraction
df['X_res'] = df['X_res'].str.replace(',', '')
df['X_res'] = df['X_res'].str.extract(r'(\d+)').astype(int)
df['Y_res'] = df['Y_res'].astype(int)

# PPI
df['ppi'] = ((df['X_res']**2 + df['Y_res']**2)**0.5 / df['Inches']).astype(float)

df.drop(columns=['ScreenResolution','Inches','X_res','Y_res'], inplace=True)

# CPU
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

def fetch_processor(text):
    if text in ['Intel Core i7','Intel Core i5','Intel Core i3']:
        return text
    elif text.split()[0] == 'Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'

df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
df.drop(columns=['Cpu','Cpu Name'], inplace=True)

#  MEMORY 
df['Memory'] = df['Memory'].astype(str).replace('\.0','', regex=True)
df['Memory'] = df['Memory'].str.replace('GB','')
df['Memory'] = df['Memory'].str.replace('TB','000')

new = df['Memory'].str.split('+', n=1, expand=True)

df['first'] = new[0].str.strip()
df['second'] = new[1]

# Layer flags
for col in ['HDD','SSD','Hybrid','Flash Storage']:
    df[f'Layer1{col.replace(" ","_")}'] = df['first'].apply(lambda x: 1 if col in x else 0)
    df[f'Layer2{col.replace(" ","_")}'] = df['second'].apply(lambda x: 1 if isinstance(x,str) and col in x else 0)

# Clean numbers (FIXED regex)
df['first'] = df['first'].str.replace(r'\D','', regex=True)
df['second'] = df['second'].fillna('0').str.replace(r'\D','', regex=True)

df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)

# Final storage
df['HDD'] = df['first']*df['Layer1HDD'] + df['second']*df['Layer2HDD']
df['SSD'] = df['first']*df['Layer1SSD'] + df['second']*df['Layer2SSD']

df.drop(columns=[col for col in df.columns if 'Layer' in col], inplace=True)
df.drop(columns=['first','second','Memory'], inplace=True)

#  GPU 
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
df = df[df['Gpu brand'] != 'ARM']
df.drop(columns=['Gpu'], inplace=True)

# OS 
def cat_os(inp):
    if inp in ['Windows 10','Windows 7','Windows 10 S']:
        return 'Windows'
    elif inp in ['macOS','Mac OS X']:
        return 'Mac'
    else:
        return 'Others/Linux'

df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'], inplace=True)

#VISUAL
sns.heatmap(df.corr(numeric_only=True))
plt.show()

sns.displot(np.log(df['Price']))

# ================= MODEL =================
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

# FIXED categorical columns
categorical_cols = ['Company','TypeName','Cpu brand','Gpu brand','os']

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), categorical_cols)
], remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ================= SAVE =================
import pickle
pickle.dump(df, open('df.pkl','wb'))
pickle.dump(pipe, open('pipe.pkl','wb'))
print("Model saved successfully")