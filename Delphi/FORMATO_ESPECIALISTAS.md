# Formato do Arquivo inputs/expert_responses.csv

## Resumo Rápido

O arquivo deve ter **exatamente estas colunas**:

```csv
expert_id,factor_source,factor_target,influence_score
```

## Exemplo Prático

```csv
expert_id,factor_source,factor_target,influence_score
1,Documentação de Projeto,Experiência da Equipe,2
1,Documentação de Projeto,Qualidade da Comunicação,3
1,Experiência da Equipe,Documentação de Projeto,4
2,Documentação de Projeto,Experiência da Equipe,1
2,Documentação de Projeto,Qualidade da Comunicação,2
2,Experiência da Equipe,Documentação de Projeto,3
```

## Regras Importantes

1. **expert_id**: Número do especialista (1, 2, 3, etc.)
2. **factor_source**: Nome EXATO do fator origem (deve coincidir com `inputs/Fatores.csv`)
3. **factor_target**: Nome EXATO do fator destino (deve coincidir com `inputs/Fatores.csv`)
4. **influence_score**: Valor de 0 a 4
   - 0 = Nenhuma influência
   - 1 = Baixa influência  
   - 2 = Média influência
   - 3 = Alta influência
   - 4 = Muito alta influência

## Conversão Automática

O sistema converte automaticamente 0-4 para 0-9:
- 0 → 0.0 | 1 → 2.25 | 2 → 4.5 | 3 → 6.75 | 4 → 9.0

## Arquivo de Exemplo

Um exemplo completo está disponível em: `inputs/expert_responses_example.csv`

## O que acontece se o arquivo não existir?

O sistema cria automaticamente dados simulados no formato correto.

## Sistema de Cache

Para economizar API durante testes:
- Na primeira execução: escolha opção 2 (fazer consultas)
- Nas próximas: escolha opção 1 (usar cache)

✅ **Pronto para usar!**