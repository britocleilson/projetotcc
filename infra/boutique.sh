#!/bin/bash

# Este script provisiona a aplicação de exemplo Online Boutique (microservices-demo)
# em um cluster Kubernetes.
# Ele assume que você tem o kubectl configurado corretamente para o seu cluster.
# Ele também assume que você baixou o repositório microservices-demo
# e está executando o script a partir do diretório raiz desse repositório,
# OU que você ajustou o CAMINHO_MANIFESTOS para apontar para o arquivo correto.

# Defina o caminho para o arquivo de manifestos do Kubernetes.
# Se você estiver executando este script a partir do diretório raiz do repositório
# microservices-demo clonado, o caminho abaixo deve funcionar.
# Caso contrário, ajuste este caminho para o local correto do arquivo kubernetes-manifests.yaml
CAMINHO_MANIFESTOS="../sample/microservices-demo/release/kubernetes-manifests.yaml"

# Verifique se o arquivo de manifestos existe
if [ ! -f "$CAMINHO_MANIFESTOS" ]; then
    echo "Erro: Arquivo de manifestos não encontrado em ${CAMINHO_MANIFESTOS}"
    echo "Certifique-se de estar executando o script a partir do diretório raiz do repositório microservices-demo"
    echo "ou ajuste a variável CAMINHO_MANIFESTOS para o caminho correto."
    exit 1
fi

echo "Aplicando manifestos da aplicação Online Boutique do arquivo: ${CAMINHO_MANIFESTOS}"

# Aplica os manifestos usando kubectl
kubectl apply -f "$CAMINHO_MANIFESTOS" -n boutique

# Verifica o status da implantação (opcional, mas útil)
echo "Verificando o status dos pods (pode levar alguns minutos para ficarem prontos)..."
kubectl get pods

echo "Provisionamento da aplicação Online Boutique iniciado."
echo "Use 'kubectl get pods' e 'kubectl get svc' para verificar o status dos componentes."




