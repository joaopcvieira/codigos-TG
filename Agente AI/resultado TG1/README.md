Esta pasta contem o resultado do TG 1. 
Para a criação deste relatório, foram consideradas as seguintes variáveis:

    Experiência da Equipe

    Disponibilidade de Tempo
    Orçamento Disponível
    Qualidade da Comunicação
    Documentação de Projeto

    Comprimento da Câmara
    Diâmetro da Câmara
    Margem Estrutural da Câmara
    Massa de Parafina
    Razão de Mistura (O/F)
    Vazão de Oxigênio
    Diâmetro do Injetor
    Pressão do Tanque
    Estabilidade da Pressão de Câmara
    Energia de Ignição


    Acabamento Superficial do Grão
    Pureza da Parafina


    Temperatura Ambiente
    Temperatura do Oxigênio

    Sucesso do Lançamento


Foi usado apenas o prompt molde: 
    PROMPT_TMPL = (
        "Considere o par de fatores abaixo no contexto do projeto de desenvolvimento de um motor foguete híbrido, cujo requisito principal de sucesso é o empuxo gerado pelo motor.\n"
        "Em uma escala de 0 a 9 ({scale}), qual o nível em que **{src}** "
        "influencia **{tgt}**? Entenda que o resultado deve ser interpretado no contexto do projeto e a influência é uma relação unidirecional. Associe com o contexto de causalidade de A em B. Responda apenas com o número."
    )


E o resultado é apresentado em duas figuras desta pasta (bem como o código usado). A matriz A obtida foi:


    A = [[0 7 7 7 7 7 7 7 5 7 7 7 7 7 7 7 5 2 3 8]
        [7 0 7 7 7 6 7 6 5 6 3 6 5 4 5 5 3 0 3 7]
        [4 7 0 3 7 7 7 5 7 7 5 7 7 4 4 5 4 0 3 7]
        [7 7 4 0 7 7 7 7 3 7 7 7 7 7 7 7 3 0 3 7]
        [7 7 5 7 0 7 7 7 5 7 7 7 7 7 7 7 3 2 3 7]
        [3 4 3 0 5 0 6 7 6 5 3 7 3 7 4 3 3 0 3 7]
        [3 4 4 2 5 7 0 7 7 7 7 7 4 7 3 3 3 0 3 7]
        [3 5 4 3 7 7 7 0 5 6 5 7 5 7 3 5 3 0 3 7]
        [3 4 4 0 3 7 7 6 0 7 3 7 3 4 4 3 3 0 3 7]
        [3 4 3 3 4 7 7 4 7 0 7 7 4 7 7 3 3 0 7 9]
        [3 4 3 0 3 7 7 6 7 7 0 7 3 7 7 3 3 0 7 9]
        [3 3 3 2 3 7 7 6 4 7 7 0 3 7 5 3 2 0 3 7]
        [3 5 3 0 7 7 7 7 6 7 7 7 0 8 5 3 3 0 3 8]
        [3 6 3 3 7 7 7 7 6 7 7 7 7 0 7 4 3 0 3 8]
        [3 4 3 2 3 5 5 3 5 6 3 7 3 7 0 3 3 0 3 8]
        [3 3 3 3 3 3 3 6 3 5 3 3 3 6 6 0 3 0 3 7]
        [3 3 3 0 3 6 5 4 7 7 3 7 3 7 7 7 0 0 3 7]
        [3 4 3 0 3 5 5 5 5 6 5 5 5 6 6 3 5 0 7 6]
        [3 3 3 0 3 7 7 6 6 7 7 7 4 7 7 3 3 3 0 7]
        [7 7 7 3 7 7 7 7 5 7 5 7 7 7 3 3 3 0 3 0]]
    
