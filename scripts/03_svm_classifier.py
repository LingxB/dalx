from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import pandas as pd


df = pd.read_csv('data/processed/seed_vectors.csv', index_col='SEED')

X = df[[str(i) for i in range(500)]].values.astype('float32')
y = df['LABELS'].values.astype('int32')


# Search for best C
svc = LinearSVC(random_state=42, max_iter=5000)
params = [
    {'C': [0.001, 0.01, 0.1, 1, 10]}
]
gs = GridSearchCV(svc, param_grid=params, n_jobs=-1, cv=3, verbose=1)
gs.fit(X, y)
print(f'best_score(acc.)={gs.best_score_:.2%}; best_param={gs.best_params_}')

c = gs.best_params_['C']

svc = LinearSVC(C=c, verbose=1, random_state=42, max_iter=5000)
svc.fit(X, y)


x = X[1,:].reshape(1, -1)

svc.predict(x)

svc.decision_function(x)



