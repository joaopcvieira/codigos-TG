# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Para estrutura de rede e aprendizado dos parâmetros
from pgmpy.estimators import HillClimbSearch, BDeuScore, MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Exemplo: Carregar o dataset "asia"
# Caso possua a biblioteca bnlearn instalada, podemos usá-la para importar o exemplo:
# try:
import bnlearn as bn
data = bn.import_example('asia')
# except ImportError:
#     # Se não estiver instalado, simulamos um dataset com as variáveis da rede ASIA (8 colunas)
#     data = pd.DataFrame({
#         "asia": np.random.choice([0, 1], size=1000),
#         "tub": np.random.choice([0, 1], size=1000),
#         "smoke": np.random.choice([0, 1], size=1000),
#         "lung": np.random.choice([0, 1], size=1000),
#         "bronc": np.random.choice([0, 1], size=1000),
#         "either": np.random.choice([0, 1], size=1000),
#         "xray": np.random.choice([0, 1], size=1000),
#         "dysp": np.random.choice([0, 1], size=1000)
#     })

# Visualizar as primeiras linhas do dataset
print("Dataset ASIA:")
print(data.head())

# Aprender a estrutura da rede utilizando Hill Climb Search e BDeu como função de pontuação
hc = HillClimbSearch(data, scoring_method=BDeuScore(data))
best_model = hc.estimate()

print("\nEstrutura da rede aprendida (arestas):")
print(best_model.edges())

# Criar o modelo bayesiano com base na estrutura aprendida
model = BayesianModel(best_model.edges())

# Aprender os CPDs (tabelas de probabilidades condicionais) com estimador de máxima verossimilhança
model.fit(data, estimator=MaximumLikelihoodEstimator)

print("\nCPDs da rede:")
for cpd in model.get_cpds():
    print(cpd)

# Plotar o DAG da rede usando networkx e matplotlib
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(model)
nx.draw(model, pos, with_labels=True, node_size=2000, node_color='lightblue', arrowsize=20)
plt.title("DAG da Rede Bayesiana")
plt.show()

# Criar o motor de inferência (equivalente ao InferenceEngine)
inference = VariableElimination(model)

# Consultar as marginais para a variável 'dysp'
marginal_dysp = inference.query(variables=['dysp'])
print("\nMarginais para 'dysp':")
print(marginal_dysp)

# Exemplo de intervenção: fixar o valor da variável 'tub' para 1 e ver o efeito em 'dysp'
marginal_dysp_intervened = inference.query(variables=['dysp'], evidence={'tub': 1})
print("\nMarginais para 'dysp' com intervenção (tub=1):")
print(marginal_dysp_intervened)

# ============================================================
# Imputação de Dados Faltantes
# ============================================================
# Para imputação de dados faltantes, podemos utilizar o IterativeImputer do scikit-learn
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Simular dados faltantes: introduzindo NaN em 10% das linhas da coluna 'dysp'
data_missing = data.copy()
missing_indices = data_missing.sample(frac=0.1, random_state=42).index
data_missing.loc[missing_indices, 'dysp'] = np.nan

# Aplicar imputação iterativa
imputer = IterativeImputer(random_state=0)
data_imputed = pd.DataFrame(imputer.fit_transform(data_missing), columns=data_missing.columns)
print("\nDataset com dados imputados:")
print(data_imputed.head())

# ============================================================
# Bootstrapping do Dataset
# ============================================================
# Para realizar bootstrapping, podemos usar a função 'resample' do scikit-learn
from sklearn.utils import resample

data_bootstrap = resample(data, n_samples=len(data), random_state=0)
print("\nExemplo de dataset bootstrapped:")
print(data_bootstrap.head())