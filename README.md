# projeto-software-com-aprendizagem-de-maquina
O código fornecido implementa uma pipeline de machine learning para analisar um conjunto de dados relacionado a câncer de pulmão e realizar predições usando diversos modelos. Aqui está uma explicação detalhada de cada parte:

---

### **1. Importação de Bibliotecas**
```python
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
```
Este trecho importa as bibliotecas necessárias para manipulação de dados (`pandas`, `numpy`), visualização (`matplotlib`) e machine learning (`sklearn`). Ele inclui ferramentas para pré-processamento, modelos de aprendizado de máquina, ensembles e métricas de avaliação.

---

### **2. Carregamento e Seleção de Dados**
```python
data = pd.read_csv('survey_lung_cancer.csv')
data = data[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
       'PEER_PRESSURE', 'CHRONIC DISEASE', 'WHEEZING',
       'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
       'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']]
```
O código lê o dataset e seleciona as colunas relevantes para a análise.

---

### **3. Codificação de Variáveis Categóricas**
```python
label_enco = LabelEncoder()

data['GENDER'] = label_enco.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = label_enco.fit_transform(data['LUNG_CANCER'])
```
As colunas categóricas `GENDER` e `LUNG_CANCER` são transformadas em valores numéricos para serem utilizadas pelos algoritmos de machine learning.

---

### **4. Análise de Correlação**
```python
correlacoes = data.corr(method='pearson')
correlacoes_com_coluna5 = correlacoes['LUNG_CANCER'][['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS','ANXIETY','CHRONIC DISEASE']]
print(correlacoes_com_coluna5.sum())
```
Este trecho calcula a correlação entre as colunas e a variável alvo (`LUNG_CANCER`). Ele soma as correlações de algumas colunas específicas com o alvo, o que pode indicar a relevância desses atributos.

---

### **5. Preparação dos Dados**
```python
target_column = 'LUNG_CANCER'
X = data.drop(columns=[target_column])
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Os dados são divididos em:
- `X`: Recursos (features).
- `y`: Variável alvo (`LUNG_CANCER`).
Em seguida, eles são divididos em conjuntos de treino e teste.

---

### **6. Identificação do Tipo de Problema**
```python
is_classification = y.nunique() <= 10
```
Se a variável alvo (`y`) possui 10 ou menos valores distintos, o problema é considerado de classificação. Caso contrário, é tratado como um problema de regressão.

---

### **7. Função para Treinamento e Avaliação**
```python
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if is_classification:
        score = accuracy_score(y_test, predictions)
        print(f"{model_name} Accuracy: {score:.2f}")
    else:
        score = mean_squared_error(y_test, predictions, squared=False)
        print(f"{model_name} RMSE: {score:.2f}")
```
Essa função treina um modelo no conjunto de treino e o avalia no conjunto de teste. 
- **Classificação:** Mede a **acurácia**.
- **Regressão:** Mede o **erro quadrático médio** (RMSE).

---

### **8. Configuração e Avaliação dos Modelos**
Dependendo do tipo de problema, são configurados diferentes modelos:

#### **Para Classificação**
```python
log_reg = LogisticRegression(max_iter=1000)
tree_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier()

voting_clf = VotingClassifier(estimators=[
    ('Logistic Regression', log_reg),
    ('Decision Tree', tree_clf),
    ('Random Forest', rf_clf),
    ('KNN', knn_clf)
], voting='hard')

train_and_evaluate(voting_clf, "Voting Classifier")
```
Os modelos individuais incluem:
- **Regressão Logística**
- **Árvore de Decisão**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

Um ensemble (`VotingClassifier`) combina esses modelos para melhorar a performance.

#### **Para Regressão**
```python
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
rf_reg = RandomForestRegressor()
knn_reg = KNeighborsRegressor()

voting_reg = VotingRegressor(estimators=[
    ('Linear Regression', lin_reg),
    ('Decision Tree', tree_reg),
    ('Random Forest', rf_reg),
    ('KNN', knn_reg)
])

train_and_evaluate(voting_reg, "Voting Regressor")
```
Os modelos incluem:
- **Regressão Linear**
- **Árvore de Decisão**
- **Random Forest**
- **KNN**

Um ensemble (`VotingRegressor`) é utilizado para combinar os modelos.

---

### **9. Salvando o Modelo Treinado**
```python
import joblib
joblib.dump(voting_clf if is_classification else voting_reg, 'especialista_em_cancer.pkl')
```
O modelo treinado é salvo em um arquivo `.pkl` para uso futuro, seja ele um `VotingClassifier` ou um `VotingRegressor`.

---

### **Resumo**
Este código é um exemplo robusto de pipeline de machine learning que pode lidar com problemas de classificação ou regressão. Ele realiza:
1. **Pré-processamento de dados**
2. **Treinamento de modelos individuais**
3. **Ensemble para combinar modelos**
4. **Avaliação de desempenho**
5. **Persistência do modelo treinado**

6. Esse código utiliza uma interface gráfica para coletar informações do usuário relacionadas a possíveis fatores de risco para câncer de pulmão. Após coletar os dados, ele usa um modelo de aprendizado de máquina previamente treinado para prever se o usuário apresenta alto risco de ter câncer de pulmão. Aqui está uma explicação detalhada:
7. ![especialista](https://github.com/user-attachments/assets/fb635321-39c7-4ace-9ca4-c9a28493a3c2)


1. **Coleta de Dados do Usuário:**
   - A interface gráfica criada com o `tkinter` apresenta campos de entrada para que o usuário forneça informações como gênero, idade, e respostas a perguntas relacionadas ao estilo de vida (como fumar, consumir álcool) e condições de saúde (como ansiedade e doenças crônicas).
   - Essas informações são convertidas em um formato adequado para o modelo (valores numéricos).

2. **Modelo de Aprendizado de Máquina:**
   - Um modelo previamente treinado e salvo como `especialista_em_cancer.pkl` é carregado usando a biblioteca `joblib`.
   - O modelo foi treinado para prever a presença ou ausência de câncer de pulmão com base nas mesmas variáveis coletadas pelo sistema.

3. **Previsão:**
   - Os dados inseridos pelo usuário são organizados em um formato esperado pelo modelo (um `DataFrame` do pandas).
   - O modelo faz uma previsão e retorna um resultado: **"YES"** (alto risco) ou **"NO"** (baixo risco).
   - A resposta é exibida em uma janela pop-up.
   -
   - ![sainda](https://github.com/user-attachments/assets/e90da42d-790a-4c1a-af04-a74378a283c8)
  
   - ![saida2](https://github.com/user-attachments/assets/22aa5ffa-3ead-44b0-8031-1aaa708b326b)



4. **Tratamento de Erros:**
   - O sistema inclui tratamento de exceções para lidar com entradas inválidas ou problemas durante a execução, garantindo que mensagens de erro informativas sejam exibidas ao usuário.

5. **Design da Interface:**
   - A interface foi desenhada para ser intuitiva e fácil de usar. Inclui:
     - Campos de entrada organizados em uma tabela com labels claros.
     - Um botão de previsão que aciona o processamento.
     - Mensagens informativas para orientar o preenchimento (por exemplo, explicando que 1 significa "Sim" e 0 significa "Não").

6. **Estilo e Identidade:**
   - O programa inclui elementos visuais, como ícones e imagens personalizadas, para melhorar a experiência do usuário e reforçar a identidade visual.

### Fluxo Geral do Usuário:
1. O usuário preenche os campos da interface com suas informações.
2. Clica no botão **Predict**.
3. Recebe o resultado da previsão em uma mensagem.

Esse sistema pode ser usado em cenários educativos ou preliminares para conscientização, mas não substitui consultas médicas ou diagnósticos profissionais.
