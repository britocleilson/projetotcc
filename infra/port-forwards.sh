#!/bin/bash

# Este script realiza o port-forwarding para os principais addons do Istio
# (Grafana, Kiali, Jaeger/Tracing, Prometheus) no namespace istio-system.
# Ele assume que você tem o kubectl configurado corretamente para o seu cluster.
# Execute este script em um terminal compatível com Bash (Git Bash, WSL, etc.).

NAMESPACE="istio-system"
NAMESPACE_APP="boutique"

echo "Iniciando port-forwarding para o app Boutique Online no namespace: ${NAMESPACE_APP}"
kubectl port-forward svc/frontend 8080:80 -n ${NAMESPACE_APP} &
FRONT_PID=$!
echo "Port forward iniciado no frontend da app com PID: ${FRONT_PID}"

echo "Iniciando port-forwarding para addons do Istio no namespace: ${NAMESPACE}"

# Port-forward para Grafana (porta 3000)
# Acessível em http://localhost:3000
echo "Iniciando port-forward para Grafana (localhost:3000)..."
kubectl port-forward svc/grafana 3000:3000 -n ${NAMESPACE} &
GRAFANA_PID=$!
echo "Port-forward para Grafana iniciado com PID: ${GRAFANA_PID}"

# Port-forward para Kiali (porta 20001)
# Acessível em http://localhost:20001
echo "Iniciando port-forward para Kiali (localhost:20001)..."
kubectl port-forward svc/kiali 20001:20001 -n ${NAMESPACE} &
KIALI_PID=$!
echo "Port-forward para Kiali iniciado com PID: ${KIALI_PID}"

# Port-forward para Jaeger/Tracing (porta 16685)
# Acessível em http://localhost:16685
# Nota: O serviço pode ser 'tracing' ou 'jaeger-query' dependendo da sua instalação.
# Com base na sua saída, o serviço é 'tracing' com porta 16685.
echo "Iniciando port-forward para Jaeger/Tracing (localhost:16685)..."
kubectl port-forward svc/tracing 16685:16685 -n ${NAMESPACE} &
JAEGER_PID=$!
echo "Port-forward para Jaeger/Tracing iniciado com PID: ${JAEGER_PID}"

# Port-forward para Prometheus (porta 9090)
# Acessível em http://localhost:9090
echo "Iniciando port-forward para Prometheus (localhost:9090)..."
kubectl port-forward svc/prometheus 9090:9090 -n ${NAMESPACE} &
PROMETHEUS_PID=$!
echo "Port-forward para Prometheus iniciado com PID: ${PROMETHEUS_PID}"

echo "Port-forwarding para todos os addons solicitados iniciado em segundo plano."
echo "Você pode acessar os dashboards nos seguintes endereços:"
echo "- Grafana: http://localhost:3000"
echo "- Kiali: http://localhost:20001"
echo "- Jaeger/Tracing: http://localhost:16685"
echo "- Prometheus: http://localhost:9090"
echo ""
echo "Para parar os processos de port-forwarding, você pode usar os PIDs listados acima"
echo "ou encontrar os processos 'kubectl port-forward' e encerrá-los."
echo "Alternativamente, fechar o terminal onde o script está rodando geralmente encerrará os processos em segundo plano."


# O script continuará rodando até que os processos em segundo plano sejam encerrados
# ou o terminal seja fechado.
# Se você quiser que o script espere (e não retorne ao prompt imediatamente),
# você pode adicionar um comando como 'wait' aqui, mas isso bloquearia o terminal.
# Para este caso, rodar em segundo plano é mais prático.


