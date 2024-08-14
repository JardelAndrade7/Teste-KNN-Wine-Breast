# Importando bibliotecas necessárias
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# A base é dividida em Treinamento (train) e Teste (test)
X,y =load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Instancia o classificador K Nearest Neighbors e realiza o "treinamento"
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Executa classificador com dados de teste e obtem resultados previstos
y_pred = neigh.predict(X_test)

# Compara os resultados previstos pelo classificados com os resultados esperados de acordo com os dados de teste
# Total: 285
# Erros: 28
soma = (y_test != y_pred).sum()
print(f"Total: {X_test.shape[0]}")
print(f"Numero erros: {soma}")