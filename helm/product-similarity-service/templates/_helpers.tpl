{{/* vim: set filetype=mustache: */}}

{{/*
Image tag — uses Chart.appVersion when values.image.tag is empty.
*/}}
{{- define "product-similarity-service.imageTag" -}}
{{- default .Chart.AppVersion .Values.image.tag -}}
{{- end -}}

{{/*
Full image reference.
*/}}
{{- define "product-similarity-service.image" -}}
{{ .Values.image.repository }}:{{ include "product-similarity-service.imageTag" . }}
{{- end -}}

{{/*
Common labels applied to all resources.
*/}}
{{- define "product-similarity-service.labels" -}}
app: product-similarity-service
app.kubernetes.io/name: product-similarity-service
app.kubernetes.io/version: {{ include "product-similarity-service.imageTag" . | quote }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
