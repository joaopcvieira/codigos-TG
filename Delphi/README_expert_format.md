# Formato do Arquivo de Dados de Especialistas

O sistema Delphi-DEMATEL aceita dados de especialistas em formato CSV. Este documento explica o formato esperado e fornece exemplos práticos.

## Formato Recomendado: Linha por Linha

**Arquivo:** `inputs/expert_responses.csv`

### Estrutura das Colunas

```csv
expert_id,factor_source,factor_target,influence_score
```

### Descrição das Colunas

- **expert_id**: Identificador único do especialista (número inteiro)
- **factor_source**: Nome do fator que exerce influência (deve coincidir com nomes em `inputs/Fatores.csv`)
- **factor_target**: Nome do fator que recebe influência (deve coincidir com nomes em `inputs/Fatores.csv`) 
- **influence_score**: Nível de influência de 0 a 4
  - 0 = Nenhuma influência
  - 1 = Baixa influência
  - 2 = Média influência
  - 3 = Alta influência
  - 4 = Muito alta influência

### Exemplo Prático

Considere os fatores: "Pressao_Combustao", "Taxa_Injecao", "Empuxo"

```csv
expert_id,factor_source,factor_target,influence_score
1,Pressao_Combustao,Taxa_Injecao,3
1,Pressao_Combustao,Empuxo,4
1,Taxa_Injecao,Pressao_Combustao,2
1,Taxa_Injecao,Empuxo,4
1,Empuxo,Pressao_Combustao,1
1,Empuxo,Taxa_Injecao,1
2,Pressao_Combustao,Taxa_Injecao,2
2,Pressao_Combustao,Empuxo,3
2,Taxa_Injecao,Pressao_Combustao,2
2,Taxa_Injecao,Empuxo,3
2,Empuxo,Pressao_Combustao,0
2,Empuxo,Taxa_Injecao,1
3,Pressao_Combustao,Taxa_Injecao,4
3,Pressao_Combustao,Empuxo,4
3,Taxa_Injecao,Pressao_Combustao,3
3,Taxa_Injecao,Empuxo,4
3,Empuxo,Pressao_Combustao,2
3,Empuxo,Taxa_Injecao,2
```

### Regras Importantes

1. **Escala**: Use sempre valores de 0 a 4 (será convertido automaticamente para 0-9 internamente)
2. **Nomes dos fatores**: Devem ser EXATAMENTE iguais aos nomes no arquivo `inputs/Fatores.csv`
3. **Auto-influência**: Não inclua pares onde factor_source = factor_target
4. **Múltiplos especialistas**: Use expert_id diferentes (1, 2, 3, etc.)
5. **Completude**: Cada especialista deve avaliar todos os pares possíveis (N×N-N pares para N fatores)

## Conversão Automática de Escala

O sistema converte automaticamente a escala dos especialistas (0-4) para a escala do LLM (0-9) usando a fórmula:

```
nova_pontuacao = (pontuacao_original / 4) * 9
```

**Mapeamento de valores:**
- 0 (nenhuma) → 0.0
- 1 (baixa) → 2.25 
- 2 (média) → 4.5
- 3 (alta) → 6.75
- 4 (muito alta) → 9.0

## Geração Automática de Dados

Se o arquivo `inputs/expert_responses.csv` não existir, o sistema criará automaticamente dados simulados com base nos fatores carregados de `inputs/Fatores.csv`.

Para usar dados reais de especialistas:
1. Crie o diretório `inputs/` se não existir
2. Crie o arquivo `expert_responses.csv` seguindo o formato acima
3. Execute o sistema normalmente

## Formato Alternativo: Matrizes Empilhadas (Legado)

O sistema também suporta um formato de matrizes empilhadas, mas o formato linha por linha é recomendado por sua simplicidade e flexibilidade.

## Verificação de Dados

Para verificar se seus dados estão corretos, execute o sistema e observe as mensagens:

- ✅ "Dados de especialistas encontrados" = Arquivo carregado com sucesso
- ⚠️ "Arquivo de especialistas não encontrado" = Sistema criará dados simulados
- 📊 "Dados de X pares de fatores carregados" = Confirmação do processamento

## Exemplo de Uso no Código

```python
# O sistema carrega automaticamente se o arquivo existir
delphi_system = DelphiDematel(
    factors=fatores,
    descriptions=descricoes,
    expert_data_path='inputs/expert_responses.csv',  # Caminho para seus dados
    memory_path="delphi_memory.json",
    provider="openai"
)
```