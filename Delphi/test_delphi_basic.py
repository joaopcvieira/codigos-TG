"""
test_delphi_basic.py
Teste básico para verificar se o sistema Delphi está funcionando corretamente.
"""

import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Adiciona o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_response_record():
    """Testa a classe ResponseRecord"""
    from delphi_dematel import ResponseRecord
    from datetime import datetime
    
    response = ResponseRecord(
        score=7,
        rationale="Justificativa técnica detalhada",
        confidence=4,
        timestamp=datetime.now().isoformat(),
        round_number=1
    )
    
    # Testa serialização
    data = response.to_dict()
    restored = ResponseRecord.from_dict(data)
    
    assert restored.score == response.score
    assert restored.rationale == response.rationale
    assert restored.confidence == response.confidence
    print("✓ ResponseRecord funcionando corretamente")

def test_expert_statistics():
    """Testa a classe ExpertStatistics"""
    from delphi_dematel import ExpertStatistics
    
    responses = [5, 6, 7, 6, 8]
    stats = ExpertStatistics(
        median=float(np.median(responses)),
        q1=float(np.percentile(responses, 25)),
        q3=float(np.percentile(responses, 75)),
        iqr=float(np.percentile(responses, 75) - np.percentile(responses, 25)),
        mean=float(np.mean(responses)),
        std=float(np.std(responses)),
        count=len(responses),
        responses=responses
    )
    
    assert stats.median == 6.0
    assert stats.count == 5
    assert len(stats.responses) == 5
    print("✓ ExpertStatistics funcionando corretamente")

def test_delphi_memory():
    """Testa a classe DelphiMemory"""
    from delphi_dematel import DelphiMemory, ResponseRecord, ExpertStatistics
    from datetime import datetime
    
    memory = DelphiMemory()
    
    # Testa adição de resposta
    response = ResponseRecord(
        score=6,
        rationale="Teste",
        confidence=3,
        timestamp=datetime.now().isoformat(),
        round_number=1
    )
    
    memory.add_response("FatorA", "FatorB", response)
    retrieved = memory.get_latest_response("FatorA", "FatorB")
    
    assert retrieved.score == 6
    assert retrieved.rationale == "Teste"
    
    # Testa estatísticas de especialistas
    stats = ExpertStatistics(
        median=5.0, q1=4.0, q3=6.0, iqr=2.0,
        mean=5.2, std=1.1, count=10, responses=[4,5,5,5,6,6,6,7,4,5]
    )
    
    memory.set_expert_stats("FatorA", "FatorB", stats)
    retrieved_stats = memory.get_expert_stats("FatorA", "FatorB")
    
    assert retrieved_stats.median == 5.0
    assert retrieved_stats.count == 10
    
    print("✓ DelphiMemory funcionando corretamente")

def test_expert_aggregator():
    """Testa a classe ExpertAggregator"""
    from delphi_dematel import ExpertAggregator
    
    # Cria matrizes de exemplo de 3 especialistas para 3 fatores
    expert1 = np.array([[0, 5, 7], [6, 0, 4], [3, 8, 0]])
    expert2 = np.array([[0, 4, 6], [7, 0, 5], [2, 7, 0]])
    expert3 = np.array([[0, 6, 8], [5, 0, 3], [4, 9, 0]])
    
    expert_matrices = [expert1, expert2, expert3]
    
    aggregated = ExpertAggregator.aggregate_expert_responses(expert_matrices)
    
    # Verifica alguns cálculos
    assert (0, 1) in aggregated  # Par (0,1) deve existir
    assert (0, 0) not in aggregated  # Diagonal não deve existir
    
    stats_01 = aggregated[(0, 1)]
    assert stats_01.count == 3
    assert stats_01.median == 5.0  # mediana de [5, 4, 6]
    
    print("✓ ExpertAggregator funcionando corretamente")

def create_test_data():
    """Cria dados de teste mínimos"""
    
    # Cria diretório de inputs se não existir
    os.makedirs('inputs', exist_ok=True)
    
    # Fatores mínimos para teste
    test_factors = pd.DataFrame({
        'fator': ['Temperatura', 'Pressao', 'Geometria'],
        'descricao': [
            'Temperatura da câmara de combustão',
            'Pressão interna da câmara',
            'Geometria do bocal'
        ]
    })
    test_factors.to_csv('inputs/Fatores.csv', index=False)
    
    # Dados de especialistas simulados
    expert_data = []
    for expert_id in range(3):
        for i in range(3):
            row = {'expert_id': f'expert_{expert_id}'}
            for j in range(3):
                if i == j:
                    row[test_factors['fator'].iloc[j]] = 0
                else:
                    # Valores aleatórios mas consistentes
                    np.random.seed(expert_id * 10 + i * 3 + j)
                    row[test_factors['fator'].iloc[j]] = np.random.randint(3, 8)
            expert_data.append(row)
    
    expert_df = pd.DataFrame(expert_data)
    expert_df.to_csv('inputs/expert_responses.csv', index=False)
    
    print("✓ Dados de teste criados")

def test_basic_delphi_initialization():
    """Testa inicialização básica do DelphiDematel"""
    
    # Mock das chamadas de API para evitar custos
    with patch('config.OPENAI_API_KEY', 'fake_key'), \
         patch('config._OPENAI_V0', False), \
         patch('openai.OpenAI') as mock_openai:
        
        # Mock do cliente OpenAI
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        from delphi_dematel import DelphiDematel
        
        # Cria dados de teste
        create_test_data()
        
        # Carrega fatores
        df_fatores = pd.read_csv('inputs/Fatores.csv')
        fatores = df_fatores['fator'].tolist()
        descricoes = df_fatores['descricao'].tolist()
        
        # Testa inicialização
        delphi_system = DelphiDematel(
            factors=fatores,
            descriptions=descricoes,
            expert_data_path='inputs/expert_responses.csv',
            memory_path='test_memory.json',
            provider='openai'
        )
        
        assert delphi_system.n == 3
        assert len(delphi_system.factors) == 3
        assert delphi_system.provider == 'openai'
        
        # Verifica se dados de especialistas foram carregados
        assert len(delphi_system.memory.expert_stats) > 0
        
        print("✓ DelphiDematel inicialização funcionando")

def main():
    """Executa todos os testes básicos"""
    print("Executando testes básicos do sistema Delphi...")
    print("=" * 50)
    
    try:
        test_response_record()
        test_expert_statistics()
        test_delphi_memory()
        test_expert_aggregator()
        test_basic_delphi_initialization()
        
        print("=" * 50)
        print("✅ Todos os testes básicos passaram!")
        print("\nO sistema está pronto para uso. Para executar o processo completo:")
        print("python example_delphi_usage.py")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Erro nos testes: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Limpa arquivos de teste
        for file in ['test_memory.json', 'inputs/Fatores.csv', 'inputs/expert_responses.csv']:
            if os.path.exists(file):
                os.remove(file)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)