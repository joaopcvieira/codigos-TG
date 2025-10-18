"""
Exemplo de uso do sistema DelphiDematel
Demonstra como usar o processo de auditoria Delphi para validar avaliações do LLM
"""

import os
import numpy as np
import pandas as pd
from delphi_dematel import DelphiDematel, ExpertAggregator

def create_sample_expert_data():
    """
    Cria dados de exemplo de especialistas para demonstração.
    Na prática, esses dados viriam de questionários reais.
    """
    # Lista de fatores (mesmo do main.py)
    df_fatores = pd.read_csv('inputs/Fatores.csv')
    fatores = df_fatores['fator'].tolist()
    n_factors = len(fatores)
    
    # Simula 5 especialistas com matrizes ligeiramente diferentes
    expert_matrices = []
    np.random.seed(42)  # Para reprodutibilidade
    
    for expert_id in range(5):
        # Cria matriz base com valores aleatórios mais conservadores que LLM
        matrix = np.random.randint(0, 8, size=(n_factors, n_factors))
        
        # Ajusta diagonal para zero
        np.fill_diagonal(matrix, 0)
        
        # Adiciona alguma variabilidade baseada no expert_id
        if expert_id == 0:  # Especialista mais conservador
            matrix = (matrix * 0.7).astype(int)
        elif expert_id == 4:  # Especialista mais liberal
            matrix = np.minimum(matrix + 1, 9)
        
        expert_matrices.append(matrix)
    
    # Salva dados em formato que pode ser lido pelo sistema
    all_data = []
    for i, matrix in enumerate(expert_matrices):
        df = pd.DataFrame(matrix, columns=fatores)
        df.insert(0, 'expert_id', f'expert_{i}')
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv('inputs/expert_responses.csv', index=False)
    
    print(f"Dados de {len(expert_matrices)} especialistas criados em 'inputs/expert_responses.csv'")
    return 'inputs/expert_responses.csv'

def demonstrate_delphi_process():
    """Demonstra o processo completo Delphi-DEMATEL"""
    
    print("="*80)
    print("DEMONSTRAÇÃO DO PROCESSO DELPHI-DEMATEL")
    print("="*80)
    
    # 1. Carrega fatores e descrições
    print(os.listdir('./inputs'))
    df_fatores = pd.read_csv('inputs/Fatores.csv')
    fatores = df_fatores['fator'].tolist()
    descricoes = df_fatores['descricao'].tolist()
    
    print(f"\nCarregados {len(fatores)} fatores para análise")
    
    # 2. Cria dados de especialistas de exemplo (remova em produção)
    expert_data_path = create_sample_expert_data()
    
    # 3. Inicializa sistema Delphi
    print("\nInicializando sistema DelphiDematel...")
    delphi_system = DelphiDematel(
        factors=fatores,
        descriptions=descricoes,
        expert_data_path=expert_data_path,
        memory_path="delphi_memory.json",
        provider="openai"  # ou "gemini"
    )
    
    # 4. Executa processo completo
    print("\nExecutando processo Delphi completo...")
    results = delphi_system.run_full_delphi_process(max_rounds=2)
    
    # 5. Mostra resultados
    print("\n" + "="*50)
    print("RESULTADOS DO PROCESSO DELPHI")
    print("="*50)
    
    print(f"\nRodadas executadas: {len(results['rounds']) + 1}")
    
    if results['final_metrics']:
        metrics = results['final_metrics']
        print(f"\nMétricas de concordância com especialistas:")
        print(f"- Diferença absoluta média: {metrics.get('mean_absolute_difference', 0):.3f}")
        print(f"- Correlação de Spearman: {metrics.get('spearman_correlation', 0):.3f}")
        print(f"- Taxa de acordo exato: {metrics.get('exact_agreement_rate', 0)*100:.1f}%")
        print(f"- Taxa de acordo próximo (±1): {metrics.get('close_agreement_rate', 0)*100:.1f}%")
    
    # 6. Gera visualizações
    print(f"\nGerando visualizações...")
    delphi_system.plot_network(title="Rede de Influência - Processo Delphi")
    
    print(f"\nProcesso concluído! Arquivos gerados:")
    print(f"- delphi_memory.json: Memória completa do processo")
    print(f"- audit_report.json: Relatório detalhado de auditoria")
    print(f"- matriz_dematel_inicial.txt: Matriz da rodada inicial")
    for i in range(len(results['rounds'])):
        print(f"- matriz_dematel_round_{i+2}.txt: Matriz da rodada {i+2}")
    
    return results

def analyze_bias_and_changes():
    """Analisa viés e mudanças no processo a partir da memória salva"""
    from delphi_dematel import DelphiMemory
    
    print("\n" + "="*50)
    print("ANÁLISE DE VIÉS E MUDANÇAS")
    print("="*50)
    
    # Carrega memória
    memory = DelphiMemory()
    memory.load_from_json("delphi_memory.json")
    
    total_pairs = len(memory.responses)
    changed_pairs = 0
    high_disagreement = 0
    
    print(f"\nAnalisando {total_pairs} pares de fatores...")
    
    disagreement_details = []
    
    for (src, tgt), responses in memory.responses.items():
        if len(responses) > 1:  # Houve mais de uma rodada
            initial_score = responses[0].score
            final_score = responses[-1].score
            
            if initial_score != final_score:
                changed_pairs += 1
                change_info = {
                    'pair': f"{src} -> {tgt}",
                    'initial_score': initial_score,
                    'final_score': final_score,
                    'change': final_score - initial_score,
                    'initial_confidence': responses[0].confidence,
                    'final_confidence': responses[-1].confidence
                }
                
                # Verifica se há dados de especialistas
                expert_stats = memory.get_expert_stats(src, tgt)
                if expert_stats:
                    change_info['expert_median'] = expert_stats.median
                    change_info['initial_diff_from_experts'] = abs(initial_score - expert_stats.median)
                    change_info['final_diff_from_experts'] = abs(final_score - expert_stats.median)
                    
                    if change_info['final_diff_from_experts'] > 2:
                        high_disagreement += 1
                        disagreement_details.append(change_info)
    
    print(f"\nResumo das mudanças:")
    print(f"- Pares que mudaram: {changed_pairs}/{total_pairs} ({changed_pairs/total_pairs*100:.1f}%)")
    print(f"- Desacordos significativos com especialistas: {high_disagreement}")
    
    if disagreement_details:
        print(f"\nMaiores desacordos (diferença > 2 pontos):")
        for detail in sorted(disagreement_details, key=lambda x: x['final_diff_from_experts'], reverse=True)[:5]:
            print(f"- {detail['pair']}: LLM={detail['final_score']}, Especialistas={detail['expert_median']:.1f}")
    
    return {
        'total_pairs': total_pairs,
        'changed_pairs': changed_pairs,
        'high_disagreement': high_disagreement,
        'disagreement_details': disagreement_details
    }

if __name__ == "__main__":
    # Garante que o diretório de trabalho está correto
    os.chdir('/Users/joaovieira/Documents/TG/codigos-TG/Delphi')
    
    try:
        # Executa demonstração completa
        results = demonstrate_delphi_process()
        
        # Analisa resultados
        bias_analysis = analyze_bias_and_changes()
        
        print(f"\n{'='*80}")
        print("PROCESSO DELPHI-DEMATEL CONCLUÍDO COM SUCESSO!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback
        traceback.print_exc()