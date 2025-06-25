


PROMETHEUS_ENDPOINT = 'http://localhost:9090/api/v1/query_range'
METRIC_QUERY = 'sum by (pod) (rate({}{{pod="{}"}}[{}]))'
GRAPH_QUERY = 'sum by (source_workload, destination_workload) (rate(istio_requests_total[{}]))'
NAMESPACE = 'boutique'

