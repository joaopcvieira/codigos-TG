# Sistema DEMATEL-Delphi

Este sistema implementa um processo de auditoria inspirado na metodologia Delphi para validar e refinar as avaliações do LLM no método DEMATEL.

## Funcionalidades Principais

### 1. Estrutura de Memória Estendida
- **ResponseRecord**: Armazena nota (0-9), justificativa textual e nível de confiança (1-5) para cada avaliação
- **DelphiMemory**: Gerencia o histórico completo de múltiplas rondas de avaliação
- **ExpertStatistics**: Agrega estatísticas de especialistas (mediana, IQR, etc.)

### 2. Agregação de Especialistas
- Carrega respostas de múltiplos especialistas
- Calcula mediana, quartis e estatísticas descritivas
- Suporta formatos CSV e numpy array

### 3. Processo Delphi
- **Rodada inicial**: LLM faz avaliações com justificativas
- **Rodadas de auditoria**: LLM reavalia considerando consenso de especialistas
- **Controle de viés**: Prompts desencorajam conformidade cega

### 4. Métricas de Concordância
- Diferença absoluta média
- Correlação de Spearman
- Correlação de Kendall Tau
- Taxa de acordos exatos e próximos

## Arquivos Principais

### `delphi_dematel.py`
Implementação principal com as seguintes classes:
- `ResponseRecord`: Registro individual de resposta
- `ExpertStatistics`: Estatísticas agregadas de especialistas
- `DelphiMemory`: Gerenciamento de memória persistente
- `ExpertAggregator`: Agregação de dados de especialistas
- `DelphiDematel`: Classe principal que implementa o processo completo

### `config.py`
Configurações centralizadas para evitar importações circulares:
- Chaves de API
- Modelos de LLM
- Templates de prompts
- Detecção de versão OpenAI

### `example_delphi_usage.py`
Demonstração completa do uso do sistema:
- Criação de dados de exemplo
- Execução do processo Delphi
- Análise de viés e mudanças

## Como Usar

### Preparação dos Dados

1. **Fatores e Descrições**: arquivo CSV com colunas `fator` e `descricao`
```csv
fator,descricao
Temperatura de Câmara,Temperatura interna da câmara de combustão
Pressão de Câmara,Pressão interna da câmara de combustão
...
```

2. **Respostas de Especialistas**: arquivo CSV com múltiplas matrizes NxN empilhadas
```csv
expert_id,fator1,fator2,fator3,...
expert_0,0,6,7,...
expert_0,7,0,8,...
expert_1,0,5,6,...
expert_1,6,0,7,...
```

### Execução do Processo

```python
from delphi_dematel import DelphiDematel

# 1. Carrega fatores
df_fatores = pd.read_csv('inputs/Fatores.csv')
fatores = df_fatores['fator'].tolist()
descricoes = df_fatores['descricao'].tolist()

# 2. Inicializa sistema
delphi_system = DelphiDematel(
    factors=fatores,
    descriptions=descricoes,
    expert_data_path='inputs/expert_responses.csv',
    memory_path='delphi_memory.json',
    provider='openai'
)

# 3. Executa processo completo
results = delphi_system.run_full_delphi_process(max_rounds=2)

# 4. Gera visualizações
delphi_system.plot_network(title="Rede de Influência - Processo Delphi")
```

### Análise dos Resultados

O processo gera vários arquivos de output:
- `delphi_memory.json`: Memória completa com todas as avaliações e justificativas
- `audit_report.json`: Relatório detalhado com métricas de concordância
- `matriz_dematel_inicial.txt`: Matriz da rodada inicial
- `matriz_dematel_round_N.txt`: Matrizes de cada rodada subsequente

## Processo Delphi Detalhado

### Rodada Inicial
1. LLM avalia cada par de fatores (i, j)
2. Fornece nota (0-9), justificativa detalhada e confiança (1-5)
3. Matriz inicial é construída e salva

### Rodadas de Auditoria
1. Para cada par, o prompt de auditoria inclui:
   - Avaliação original do LLM (nota, justificativa, confiança)
   - Estatísticas dos especialistas (mediana, IQR, contagem)
   - Instrução para manter autonomia técnica
2. LLM pode manter ou alterar avaliação com nova justificativa
3. Mudanças são registradas e analisadas

### Controle de Viés
- Prompts explicitamente desencorajam seguir a maioria
- Enfase em argumentos técnicos substantivos
- Registro de desacordos com especialistas e motivos

## Métricas e Análises

### Concordância com Especialistas
- **MAD** (Mean Absolute Difference): Diferença média absoluta
- **Correlação de Spearman**: Correlação de postos
- **Taxa de Acordo**: Proporção de avaliações idênticas ou próximas

### Análise de Mudanças
- Quantas avaliações foram alteradas entre rodadas
- Magnitude das mudanças
- Relação entre mudanças e consenso de especialistas

### Análise de Confiança
- Distribuição dos níveis de confiança
- Relação entre confiança e concordância com especialistas
- Evolução da confiança ao longo das rodadas

## Configuração de Ambiente

### Variáveis de Ambiente (.env)
```bash
OPENAI_API_KEY=sua_chave_openai
GEMINI_API_KEY=sua_chave_gemini  # opcional
LLM_PROVIDER=openai  # ou "gemini"
OPENAI_ORG_ID=sua_organizacao  # opcional
```

### Dependências Python
```bash
pip install openai numpy pandas networkx plotly scipy python-dotenv
pip install google-generativeai  # opcional, para Gemini
```

## Exemplo de Saída

```
RESULTADOS DO PROCESSO DELPHI
==================================================

Rodadas executadas: 3

Métricas de concordância com especialistas:
- Diferença absoluta média: 1.234
- Correlação de Spearman: 0.756
- Taxa de acordo exato: 23.4%
- Taxa de acordo próximo (±1): 67.8%

Resumo das mudanças:
- Pares que mudaram: 12/132 (9.1%)
- Desacordos significativos com especialistas: 5

Maiores desacordos (diferença > 2 pontos):
- Temperatura -> Pressão: LLM=8, Especialistas=5.5
- Geometria -> Empuxo: LLM=3, Especialistas=6.0
```

## Extensões Futuras

1. **Múltiplos LLMs**: Comparar diferentes modelos no processo Delphi
2. **Pesos Adaptativos**: Dar maior peso a especialistas com maior concordância
3. **Análise Temporal**: Estudar como as avaliações evoluem ao longo do tempo
4. **Interface Gráfica**: Dashboard para visualizar o processo em tempo real

## Limitações

1. Requer dados de especialistas para auditoria completa
2. Processo pode ser demorado para matrizes grandes
3. Qualidade depende da clareza dos prompts e descrições dos fatores
4. Custos de API podem ser significativos para múltiplas rodadas