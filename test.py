import pandas as pd
import numpy as np

data = {
    'survived': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'pclass':   [1, 3, 1, 3, 2, 3, 1, 2, 3, 1,
                 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
    'age':      [22, 35, np.nan, 27, 14, np.nan, 58, 20, 33, 45,
                 29, np.nan, 61, 18, 24, 37, np.nan, 29, 41, 16],
    'fare':     [150, 7.5, 200, 8.0, 21, 7.2, 300, 13, 8.5, 250,
                 18, 7.8, 350, 16, 9.0, 275, 19, 8.2, 400, 17],
    'sex':      ['male','male','female','female','female','male',
                 'female','male','male','female','female','male',
                 'female','male','male','female','female','male','female','male'],
    'embarked': ['S','S','C','S','C','Q','C','S','Q','C',
                 'S','Q','C','S','S','C','Q','S','C','S']
}

df = pd.DataFrame(data)

CATEGORIES = df.columns[[1,4,5]].values
NUMS = df.columns[[2,3]].values

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

train, test = train_test_split(df, test_size=.2, shuffle=True, random_state = 42, stratify=df.sex)

num_pipe = Pipeline([
    ("median", SimpleImputer(strategy="median")),
    ("scalar", StandardScaler())
])

cat_pipe = Pipeline([
    ("onehot", OneHotEncoder(sparse_output=False))
    ])

full_pipe = ColumnTransformer([
    ("numeric_cols", num_pipe, NUMS),
    ("categorical_cols", cat_pipe, CATEGORIES)
    ])

X = full_pipe.fit_transform(train.drop("survived", axis=1))
y=train['survived']

lin_reg = LinearRegression()
lin_reg.fit(X=X,y=y)

Xtest = full_pipe.transform(test.drop("survived", axis=1))
ytest = test['survived']
predictions = lin_reg.predict(X=Xtest)