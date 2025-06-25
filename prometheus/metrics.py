import json
from typing import Dict, List, Tuple
import requests
import env
from kubernetes import client, config

# Função utilizada para obter os nomes dos PODs, excluindo o loadgenetrator
def get_pods_simplified() -> List[Tuple[str, str]]:
    """
    Obtém informações de pods de um namespace do Kubernetes, excluindo 'loadgenerator'.

    Carrega a configuração do Kubernetes (kubeconfig local ou in-cluster),
    lista os pods no namespace especificado por `env.NAMESPACE`, e retorna
    uma lista de tuplas `(nome_do_pod, nome_do_serviço)`, excluindo pods
    cujo nome começa com 'loadgenerator-'.

    O nome do serviço é determinado pelas labels 'app' ou 'component',
    ou pela primeira parte do nome do pod.

    Returns:
        Uma lista de tuplas, onde cada tupla contém o nome do pod e o nome
        do serviço associado. Ex: [('frontend-abcde', 'frontend'), ('currencyservice-fghij', 'currencyservice')]

    Raises:
        kubernetes.client.exceptions.ApiException: Se houver um erro ao
                                                 interagir com a API do Kubernetes.
        FileNotFoundError: Se o kubeconfig local não for encontrado
                           (ao usar load_kube_config).
        Exception: Outras exceções que possam ocorrer durante o carregamento
                   da configuração ou interação com a API.
    """
    namespace = env.NAMESPACE
    # Carrega configuração (usa kubeconfig local ou in-cluster)
    # Use config.load_incluster_config() dentro do cluster
    config.load_kube_config()

    v1 = client.CoreV1Api()

    # Lista os pods em um namespace específico
    ret = v1.list_namespaced_pod(namespace)

    # Usa list comprehension para filtrar e formatar os dados em uma linha
    pod_info = [
        (
            pod.metadata.name,
            pod.metadata.labels.get("app") or pod.metadata.labels.get("component") or pod.metadata.name.split("-")[0]
        )
        for pod in ret.items
        if pod.metadata.name.split("-")[0] != 'loadgenerator'
    ]

    return pod_info

# Função para buscar dados do Prometheus utilizando uma consulta, intervalo de tempo e resolução
def get_prometheus_data(query: str, start: int, end: int, resolution: int) -> Dict:
    """
    Busca dados de série temporal do Prometheus usando a API query_range.

    Args:
        query: A string da consulta Prometheus.
        start: O timestamp de início (inclusive) para o intervalo da consulta.
        end: O timestamp de fim (inclusive) para o intervalo da consulta.
        resolution: A resolução de tempo (passo) em segundos entre os pontos de dados.

    Returns:
        Um dicionário contendo os resultados da consulta do Prometheus.

    Raises:
        requests.exceptions.HTTPError: Se a requisição HTTP para o Prometheus falhar.
    """
    response = requests.get(env.PROMETHEUS_ENDPOINT, {
        "query": query, "start": start, "end": end, "step": resolution
    })
    response.raise_for_status()  # Levanta erro caso a requisição falhe (status code 4xx ou 5xx).
    response = json.loads(response.text)  # Converte a resposta JSON para um dicionário Python.
    return response["data"]["result"] # Retorna a parte dos resultados dos dados.

def get_prometheus_metrics(metric: str, pod: str, step: str, start: int, end: int, resolution: int) -> Dict:
    """
    Recupera dados de uma métrica específica do Prometheus para um determinado pod
    dentro de um intervalo de tempo.

    Esta função formata uma query do Prometheus utilizando a métrica, o nome do pod
    e o passo (intervalo) desejado, e então chama a função interna
    `get_prometheus_data` para executar a query e buscar os dados.

    Args:
        metric (str): O nome da métrica do Prometheus a ser consultada (ex: 'cpu_usage_seconds_total').
        pod (str): O nome ou identificador do pod para o qual a métrica será buscada.
        step (str): O passo ou intervalo de tempo entre os pontos de dados retornados
                    (ex: '1m' para 1 minuto, '5m' para 5 minutos).
        start (int): O timestamp Unix (em segundos) do início do intervalo de tempo da consulta.
        end (int): O timestamp Unix (em segundos) do final do intervalo de tempo da consulta.
        resolution (int): A resolução dos dados. Este parâmetro é passado diretamente
                          para `get_prometheus_data` e seu significado exato pode
                          depender da implementação dessa função (comumente relacionado
                          ao ajuste automático do passo ou à densidade dos pontos de dados).

    Returns:
        Dict: Um dicionário contendo os dados da métrica retornados pela função
              `get_prometheus_data`. O formato exato do dicionário depende da
              resposta da API do Prometheus e como `get_prometheus_data` a processa.

    Raises:
        Pode levantar exceções dependendo da implementação de `get_prometheus_data`,
        como erros de conexão, erros na query do Prometheus, ou erros de parsing
        da resposta.

    Observações:
        - Esta função depende da existência de uma variável de ambiente ou configuração
          chamada `env.METRIC_QUERY` que deve ser uma string formatável contendo
          placeholders para `metric`, `pod` e `step`.
        - Esta função depende da existência e correta implementação da função
          `get_prometheus_data`, que é responsável por executar a query formatada
          no Prometheus e retornar os dados.
    """
    # A variável env.METRIC_QUERY deve ser uma string formatável, por exemplo:
    # env.METRIC_QUERY = 'sum(rate({}[pod="{"]{})) by (pod)'
    query = env.METRIC_QUERY.format(metric, pod, step)
    return get_prometheus_data(query, start, end, resolution)

def get_prometheus_graph(step: str, start: int, end: int, resolution: int) -> Dict:
    """
    Recupera dados de uma query predefinida do Prometheus para ser utilizada em gráficos,
    dentro de um intervalo de tempo.

    Esta função utiliza um template de query definido em `env.GRAPH_QUERY`, formatando-o
    apenas com o passo (intervalo) desejado. Em seguida, chama a função interna
    `get_prometheus_data` para executar a query e buscar os dados necessários para
    visualização em gráficos.

    Args:
        step (str): O passo ou intervalo de tempo entre os pontos de dados retornados
                    (ex: '1m' para 1 minuto, '5m' para 5 minutos). Este valor é usado
                    para formatar o template da query e também é passado para
                    `get_prometheus_data`.
        start (int): O timestamp Unix (em segundos) do início do intervalo de tempo da consulta.
        end (int): O timestamp Unix (em segundos) do final do intervalo de tempo da consulta.
        resolution (int): A resolução dos dados. Este parâmetro é passado diretamente
                          para `get_prometheus_data` e seu significado exato pode
                          depender da implementação dessa função (comumente relacionado
                          ao ajuste automático do passo ou à densidade dos pontos de dados).

    Returns:
        Dict: Um dicionário contendo os dados retornados pela função
              `get_prometheus_data`. O formato exato do dicionário depende da
              resposta da API do Prometheus e como `get_prometheus_data` a processa,
              mas espera-se que seja adequado para a construção de gráficos.

    Raises:
        Pode levantar exceções dependendo da implementação de `get_prometheus_data`,
        como erros de conexão, erros na query do Prometheus, ou erros de parsing
        da resposta.

    Observações:
        - Esta função depende da existência de uma variável de ambiente ou configuração
          chamada `env.GRAPH_QUERY` que deve ser uma string formatável contendo
          um placeholder para o `step`.
        - Esta função depende da existência e correta implementação da função
          `get_prometheus_data`, que é responsável por executar a query formatada
          no Prometheus e retornar os dados.
    """
    # A variável env.GRAPH_QUERY deve ser uma string formatável com um placeholder para o step, por exemplo:
    # env.GRAPH_QUERY = 'sum(node_memory_usage_bytes{}) by (instance)' # Exemplo simples que não usa step na query string diretamente, mas pode ser formatado
    # Ou, mais comum para gráficos com range query:
    # env.GRAPH_QUERY = 'sum(rate(container_cpu_usage_seconds_total{})) by (container)' # O step é mais relevante na chamada de get_prometheus_data
    # Neste caso específico, a formatação parece aplicar o step diretamente na string da query, o que é menos comum para queries de gráfico com range:
    # env.GRAPH_QUERY = 'sua_query_aqui[{}]' # Onde {} será substituído pelo step
    query = env.GRAPH_QUERY.format(step) # Isso implica que env.GRAPH_QUERY contém um placeholder para o step na string da query.
    return get_prometheus_data(query, start, end, resolution)
