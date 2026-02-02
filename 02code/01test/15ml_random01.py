import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


wine = pd.read_csv("data/wine.csv")

data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=42)

rf = RandomForestClassifier(n_jobs=-1,random_state=42)
# n_estimators=100 결정트리개수 기본값