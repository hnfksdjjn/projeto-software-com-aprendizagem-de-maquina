import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

data = pd.read_csv('survey_lung_cancer.csv')
data = data[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
       'PEER_PRESSURE', 'CHRONIC DISEASE', 'WHEEZING',
       'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
       'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']]


label_enco = LabelEncoder()

x = data.iloc[:,0:16].values

data['GENDER'] = label_enco.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = label_enco.fit_transform(data['LUNG_CANCER'])


correlacoes = data.corr(method='pearson')
correlacoes_com_coluna5 = correlacoes['LUNG_CANCER'][['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS','ANXIETY','CHRONIC DISEASE']]
print(correlacoes_com_coluna5.sum())

# Carregar o dataset (substitua 'your_dataset.csv' pelo caminho do seu arquivo)
data = data

# Separar os dados em X (recursos) e y (alvo)
# Substitua 'target_column' pelo nome da coluna alvo
target_column = 'LUNG_CANCER'
X = data.drop(columns=[target_column])
y = data[target_column]

# Dividir o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identificar se o problema é de regressão ou classificação
is_classification = y.nunique() <= 10  # Heurística para problemas de classificação

# Função para treinar e avaliar modelos
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if is_classification:
        score = accuracy_score(y_test, predictions)
        print(f"{model_name} Accuracy: {score:.2f}")
    else:
        score = mean_squared_error(y_test, predictions, squared=False)
        print(f"{model_name} RMSE: {score:.2f}")

# Modelos individuais
if is_classification:
    log_reg = LogisticRegression(max_iter=1000)
    tree_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier()
    knn_clf = KNeighborsClassifier()

    # Ensemble usando VotingClassifier
    voting_clf = VotingClassifier(estimators=[
        ('Logistic Regression', log_reg),
        ('Decision Tree', tree_clf),
        ('Random Forest', rf_clf),
        ('KNN', knn_clf)
    ], voting='hard')

    train_and_evaluate(voting_clf, "Voting Classifier")
else:
    lin_reg = LinearRegression()
    tree_reg = DecisionTreeRegressor()
    rf_reg = RandomForestRegressor()
    knn_reg = KNeighborsRegressor()

    # Ensemble usando VotingRegressor
    voting_reg = VotingRegressor(estimators=[
        ('Linear Regression', lin_reg),
        ('Decision Tree', tree_reg),
        ('Random Forest', rf_reg),
        ('KNN', knn_reg)
    ])

    train_and_evaluate(voting_reg, "Voting Regressor")
    
import joblib
joblib.dump(voting_clf if is_classification else voting_reg, 'especialista_em_cancer.pkl')

