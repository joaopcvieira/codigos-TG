# Sistema de Cache Granular - Demonstração

## Visão Geral

O sistema agora oferece controle granular sobre o cache, permitindo escolher individualmente para cada etapa:

1. **Rodada Inicial** - Matriz base gerada pelo LLM
2. **Rodada 2** - Primeira auditoria Delphi 
3. **Rodada 3** - Segunda auditoria Delphi

## Como Funciona

### Antes de cada rodada, você verá:

```
======================================================================
CONTROLE DE CACHE - RODADA INICIAL
======================================================================
✅ Cache encontrado para Rodada Inicial
   Data: 2024-10-18 14:30:25
   Descrição: Rodada Inicial - Matriz Base
   Matriz: 12x12 elementos
   Justificativas: 132 pares de fatores

🔄 Opções para Rodada Inicial:
1. 📂 Usar cache salvo (RECOMENDADO - economiza API)
2. 🆕 Fazer novas consultas ao LLM (sobrescreve cache desta rodada)

Escolha para Rodada Inicial (1 ou 2): 
```

### Vantagens do Sistema Granular

- ✅ **Economia de API**: Use cache para etapas já validadas
- ✅ **Flexibilidade**: Refaça apenas etapas específicas  
- ✅ **Debugging**: Teste diferentes cenários facilmente
- ✅ **Iteração**: Ajuste uma rodada sem perder o trabalho das outras

### Estrutura do Cache

O cache é salvo em `llm_results_cache.json`:

```json
{
  "timestamp": "2024-10-18 14:30:25",
  "description": "Cache de resultados LLM", 
  "rounds": [
    {
      "round_number": 0,
      "description": "Rodada Inicial - Matriz Base",
      "matrix": [[...]], 
      "justifications": {...},
      "timestamp": "2024-10-18 14:30:25"
    },
    {
      "round_number": 1,
      "description": "Rodada 2 - Auditoria Delphi",
      "matrix": [[...]],
      "justifications": {...},
      "audit_metrics": {...},
      "timestamp": "2024-10-18 14:35:12"  
    }
  ]
}
```

## Cenários de Uso

### 1. Desenvolvimento/Teste
- Use cache para todas as rodadas após a primeira execução
- Economiza significativamente o consumo de API

### 2. Ajuste de Parâmetros
- Use cache para rodadas validadas
- Refaça apenas a rodada que precisa de ajuste

### 3. Análise de Convergência  
- Execute rodada inicial do cache
- Teste diferentes estratégias nas rodadas de auditoria

### 4. Primeira Execução
- Todas as rodadas farão consultas ao LLM
- Cache será criado automaticamente

## Comandos Úteis

Para limpar o cache completamente:
```bash
rm llm_results_cache.json
```

Para ver o conteúdo do cache:
```python
import json
with open('llm_results_cache.json') as f:
    cache = json.load(f)
    print(f"Rodadas: {len(cache['rounds'])}")
```

## Próximos Passos

1. Execute `python example_delphi_usage.py`
2. Na primeira vez, escolha opção 2 para todas as rodadas
3. Nas execuções seguintes, escolha opção 1 para economizar API
4. Use opção 2 apenas quando quiser refazer uma rodada específica

🎯 **O sistema está pronto para uso com controle total sobre o cache!**