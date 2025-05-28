import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
dataset='IMDB Dataset.csv'
dataset = pd.read_csv(dataset)
vector = TfidfVectorizer(max_features = 10000)
Label = LabelEncoder()
x = vector.fit_transform(dataset['review'])
y = Label.fit_transform(dataset['sentiment'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 0)
model = RandomForestClassifier(n_estimators = 100,n_jobs=-1)
model.fit(x_train,y_train)

data =input('ป้อนมา : ')
data_t=vector.transform([data])
scores = model.predict(data_t)
if scores == 1:
  print('บวก')
elif scores == 0:
  print('ลบ')
