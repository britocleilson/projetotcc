#!/bin/bash
set -e

echo "Instalando ISTIO. considere que o download ja foi realizado e adicionado ao path de forma correta"

istioctl install --set profile=demo -y

echo "criando o namespace..."
kubectl create namespace boutique

echo "injetando o sidecar no namespace"
kubectl label namespace boutique istio-injection=enabled




echo "Subindo os add-ons do Istio..."


echo "Aplicando Prometheus..."
kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/addons/prometheus.yaml


echo "Aplicando Kiali..."
kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/addons/kiali.yaml


echo "Aplicando Jaeger..."
kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/addons/jaeger.yaml


echo "Aplicando Grafana..."
kubectl apply -f https://raw.githubusercontent.com/istio/istio/master/samples/addons/grafana.yaml


echo "Todos os add-ons foram aplicados. Aguarde alguns instantes para que os pods sejam iniciados."

