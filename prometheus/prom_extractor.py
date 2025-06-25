from tqdm import tqdm
from tqdm.contrib.itertools import product
import metrics as mp
import pandas as pd
import numpy as np
import os
import env
from datetime import datetime, timedelta

# Define o tempo final (agora) e o tempo inicial (24 horas atrás)
end_time = datetime.now() # Obtém a data e hora atuais.
start_time = end_time - timedelta(hours=24) # Calcula a data e hora 24 horas antes do tempo final.

# Converte para timestamp (segundos desde Epoch), que o Prometheus usa
END_TIMESTAMP = int(end_time.timestamp()) # Converte o objeto datetime do tempo final para um timestamp inteiro.
START_TIMESTAMP = int(start_time.timestamp()) # Converte o objeto datetime do tempo inicial para um timestamp inteiro.

# Constantes da configuração
RESOLUTION_STEP = 300         # 900s = 15m | Passo de resolução da consulta (em segundos). Define a granularidade dos dados retornados.
RATE_STEP = "1h"#24           # Janela de tempo usada no cálculo de taxa (por exemplo, 'rate(metric[2h])').
OUTPUT_DIR = "../output"      # Diretório onde os arquivos de saída (CSVs, NPZs) serão salvos.
WINDOW_SIZE = 12              # Número de passos de tempo anteriores a serem considerados para formar a janela de entrada para modelos de previsão.
HORIZONT = 1                  # Horizonte de predição (nº de passos de tempo à frente). O valor que queremos prever.
STATS_DIR = None              # Diretório contendo estatísticas (média e desvio padrão) para normalização. Se None, as estatísticas são calculadas a partir dos dados atuais.
IS_TEST = False               # Flag indicando se o script está rodando em modo de teste. (Não tem efeito no fluxo atual do script)


# Definindo as métricas que serão obtidas
metric_info = [
        ("container_cpu_usage_seconds_total", "cpu"),
        ("container_memory_usage_bytes", "mem")
]

# Definindo de quais pods serão obtidas as metricas
# Neste caso serão de todos os pods, exceto o de loadbalance
pod_info = mp.get_pods_simplified()

# criando o diretório de saída caso não exista
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Carregando métricas dos pods a partir do Prometheus...")
node_df = [] # Lista para armazenar DataFrames contendo dados de métricas por pod.

for (metric_id, metric_name),(pod_id, pod_name) in tqdm(product(metric_info, pod_info)):

    # Busca os dados da métrica atual para o pod atual.
    result = mp.get_prometheus_metrics(metric_id, pod_id, RATE_STEP, START_TIMESTAMP, END_TIMESTAMP, RESOLUTION_STEP)

    if not result:  # Se a consulta não retornar resultados para esta combinação, pula para a próxima.
        continue
    values = result[0]["values"]  # Extrai os valores da série temporal.

    # Cria um DataFrame pandas a partir dos valores. A primeira coluna é timestamp, a segunda é o valor da métrica
    # nomeada como "nome_do_pod-nome_da_metrica". Define o timestamp como índice.
    df = pd.DataFrame(values, columns=["timestamp", f"{pod_name}-{metric_name}"]).set_index("timestamp")
    node_df.append(df)  # Adiciona o DataFrame criado à lista.

# Concatena todos os DataFrames na lista ao longo do eixo das colunas (axis=1).
# O join="inner" garante que apenas timestamps presentes em *todos* os DataFrames sejam mantidos.
# apply(pd.to_numeric) tenta converter todos os dados no DataFrame resultante para tipos numéricos.
node_df = pd.concat(node_df, axis=1, join="inner").apply(pd.to_numeric)

# Se nenhum diretório de estatísticas foi fornecido, calcula e salva as estatísticas de normalização.
if STATS_DIR is None:
    node_mean = node_df.mean()  # Calcula a média de cada coluna (métrica).
    node_std = node_df.std()  # Calcula o desvio padrão de cada coluna (métrica).

    # Salva a média e o desvio padrão em arquivos CSV no diretório de saída.
    node_mean.to_csv(os.path.join(OUTPUT_DIR, "pod_mean.csv"))
    node_std.to_csv(os.path.join(OUTPUT_DIR, "pod_std.csv"))
    print("Salvando estatísticas de normalização...")

    # Normaliza os dados do nó (z-score normalization): (valor - média) / desvio_padrao.
    node_dfs = (node_df - node_mean) / node_std

#Criando os arquivos de métricas e requests
print("Carregando informações de requisições entre pods...")
# Busca os dados do grafo (taxas de requisição entre workloads).
graph_query = mp.get_prometheus_graph(RATE_STEP, START_TIMESTAMP, END_TIMESTAMP, RESOLUTION_STEP)
graph_df = [] # Lista para armazenar os dados das arestas do grafo.

# Processa os resultados da consulta do grafo.
for result in tqdm(graph_query):
    source = result["metric"].get("source_workload")  # Obtém o workload de origem da aresta.
    dest = result["metric"].get("destination_workload")  # Obtém o workload de destino da aresta.

    # Ignora arestas que envolvem workloads 'unknown' ou 'loadgenerator'.
    if source == "unknown" or dest == "unknown" or source == "loadgenerator" or dest == "loadgenerator":
        continue
    # Itera sobre os pontos de dados da série temporal para a aresta atual.
    for [timestamp, value] in result["values"]:
        # Adiciona os dados da aresta (timestamp, origem, destino, valor) à lista. Converte o valor para float.
        graph_df.append([timestamp,source,dest,float(value)])

# Cria um DataFrame pandas a partir dos dados das arestas.
# value = taxa de crescimento por segundo do contador dentro do intervalo.
graph_df = pd.DataFrame(graph_df, columns=["timestamp", "from", "to", "value"])

# Filtra o DataFrame do grafo para incluir apenas timestamps que também estão presentes no DataFrame de nós.
# Define o timestamp como índice.
graph_df = graph_df[graph_df["timestamp"].isin(node_df.index)].set_index("timestamp")
graph_df["value"] = graph_df["value"].astype(float) # Garante que a coluna de valor seja do tipo float.

# Ordena ambos os DataFrames pelo índice (timestamp).
node_df = node_df.sort_values("timestamp")
graph_df = graph_df.sort_values("timestamp")

# Salva os DataFrames processados de nós e arestas em arquivos CSV no diretório de saída.
node_df.to_csv(os.path.join(OUTPUT_DIR, "pod_metrics.csv"))
graph_df.to_csv(os.path.join(OUTPUT_DIR, "pod_requests.csv"))

# Gerando os arquivos NPZ contendo os features e os targets
print("Gerando dataset de features de nós (pods)...")
X_node = [] # Lista para armazenar as sequências de features de entrada para a previsão de nós.
y_node = [] # Lista para armazenar os valores alvo para a previsão de nós.

# Gera as janelas de tempo (features de entrada) e os valores correspondentes no horizonte (alvos).
# Itera desde o final da primeira janela de entrada possível até o último ponto que ainda permite
# calcular um horizonte de predição válido.
for i in tqdm(range(WINDOW_SIZE, len(node_df) - HORIZONT + 1)):
    # Extrai a janela de dados de nós (WINDOW_SIZE passos anteriores) para o timestamp atual.
    node_window = node_df.iloc[i - WINDOW_SIZE: i]
    # Extrai os valores dos nós no horizonte de predição (HORIZONT passos à frente do final da janela).
    node_horizont = node_df.iloc[i + HORIZONT - 1]

    x, y = [], []  # Listas temporárias para as features e alvos do passo de tempo atual.

    # Itera através de cada pod para extrair seus dados de CPU e memória na janela e no horizonte.
    for (_, pod_name) in pod_info:
        # Extrai os dados de 'cpu' e 'mem' para o pod atual na janela de tempo e converte para numpy array.
        x.append(node_window[[f"{pod_name}-cpu", f"{pod_name}-mem"]].to_numpy())
        # Extrai os dados de 'cpu' e 'mem' para o pod atual no ponto do horizonte e converte para numpy array.
        y.append(node_horizont[[f"{pod_name}-cpu", f"{pod_name}-mem"]].to_numpy())

    # Converte a lista de arrays 'x' para um array numpy e troca os eixos para ter a forma [timesteps, pods, metrics].
    # Adiciona ao X_node.
    X_node.append(np.array(x).swapaxes(0, 1))
    # Converte a lista de arrays 'y' para um array numpy. Adiciona ao y_node.
    y_node.append(np.array(y))

# Converte as listas de arrays X_node e y_node em arrays numpy finais.
X_node = np.array(X_node)
y_node = np.array(y_node)

# Salva os arrays de features de nós (X) e alvos (y) em um arquivo numpy compactado (.npz).
np.savez(os.path.join(OUTPUT_DIR, "node_features.npz"), X=X_node, y=y_node)
# Imprime as formas dos datasets de nós gerados.
print("Shape do dataset de nós: X =", X_node.shape, ", y =", y_node.shape)

print("Gerando dataset de arestas (comunicação entre pods)...")
# Agrupa o DataFrame de arestas por timestamp para facilitar o acesso aos dados de aresta para cada ponto no tempo.
edge_timed = graph_df.groupby("timestamp")
A_graph = [] # Lista para armazenar as sequências de matrizes de adjacência (representando o estado do grafo ao longo do tempo).
graph = [] # Lista temporária para construir uma sequência de matrizes de adjacência para a janela atual.

# Gera as sequências de matrizes de adjacência que representam o estado do grafo ao longo do tempo.
# Itera até o ponto onde uma janela completa (WINDOW_SIZE) de dados de grafo pode ser formada antes do horizonte.
for k in tqdm(range(len(node_df) - HORIZONT)):
    node = node_df.iloc[k] # Obtém a linha do DataFrame de nós para o timestamp atual (usado principalmente para obter o timestamp).

    # Obtém as informações das arestas para o timestamp atual a partir do groupby.
    edges_info = edge_timed.get_group(node.name)

    # Inicializa uma matriz de adjacência (A) com zeros. O tamanho é (número de pods) x (número de pods).
    A = np.zeros((len(pod_info), len(pod_info)))

    # Preenche a matriz de adjacência com os valores de requisição entre os pods.
    for i in range(len(pod_info)):
        for j in range(len(pod_info)):
            from_pod = pod_info[i][1]  # Obtém o nome do pod de origem com base no índice i.
            to_pod = pod_info[j][1]  # Obtém o nome do pod de destino com base no índice j.

            # Consulta as informações das arestas obtidas para o timestamp atual para encontrar a aresta entre from_pod e to_pod.
            query = edges_info.query("`from` == @from_pod and `to` == @to_pod")

            # Se a consulta retornar resultados (ou seja, se existir uma aresta entre esses pods no timestamp atual)...
            if len(query) > 0:
                A[i, j] = query["value"].item()  # Define o valor na matriz de adjacência como o valor da requisição.

    graph.append(A) # Adiciona a matriz de adjacência gerada para o timestamp atual à lista temporária 'graph'.

    # Se a lista temporária 'graph' atingir o tamanho da janela especificada...
    if len(graph) == WINDOW_SIZE:
        A_graph.append(graph) # Adiciona a sequência de matrizes de adjacência (a janela) à lista principal A_graph.
        graph = graph[1:] # Desliza a janela: remove a matriz de adjacência mais antiga da lista temporária.


# Converte a lista de sequências de matrizes de adjacência em um array numpy final.
A_graph = np.array(A_graph)
# Salva o array de features de arestas (as sequências de matrizes de adjacência) em um arquivo numpy compactado (.npz).
np.savez(os.path.join(OUTPUT_DIR, "edge_features.npz"), A=A_graph)
# Imprime a forma do dataset de arestas gerado.
print("Shape do dataset de arestas:", A_graph.shape)
#print(A_graph)










