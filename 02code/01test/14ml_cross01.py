import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold



wine = pd.read_csv("data/wine.csv")
# print(wine.info())
# print(wine.describe())

data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()


# train : test = 80 : 20
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=42)
# print(X_train.shape,X_test.shape)

# 검증세트 (validation set) 
X_subtrain, X_val,y_subtrain,y_val = train_test_split(X_train,y_train,test_size=0.2, random_state=42)
# print(X_subtrain.shape,X_val.shape)

dt = DecisionTreeClassifier(random_state=42) # class 적용
dt.fit(X_subtrain,y_subtrain)

# print(dt.score(X_subtrain,y_subtrain))
# print(dt.score(X_val,y_val))



#교차검증 (cross-validation)

scores = cross_validate(dt,X_train,y_train) # cv=5 기본값
print(scores)

#fit_time:모델훈련시간 / score_time:모델예측시간 / test_score:모델성능

print(f'교차검증1 점수 : {np.mean(scores['test_score'])}')



scores = cross_validate(dt,X_train,y_train,cv=StratifiedKFold())
print(f'교차검증2 점수 : {np.mean(scores['test_score'])}')


splitter = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
scores = cross_validate(dt,X_train,y_train,cv=splitter)
print(f'교차검증3 점수 : {np.mean(scores['test_score'])}')


dt.fit(X_train,y_train)
print(dt.score(X_test,y_test))