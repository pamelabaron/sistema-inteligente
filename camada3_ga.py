import random
import math


random.seed(42)

# catalogo de produtos disponiveis - simulado
# cada produto tem: nome, preco, categoria, pontuacao_media_historica
CATALOGO_PRODUTOS = [
    {"id": 0,  "nome": "smartphone basico",       "preco": 599,  "categoria": "eletronicos",  "pontuacao_historica": 0.72},
    {"id": 1,  "nome": "smartphone premium",      "preco": 3499, "categoria": "eletronicos",  "pontuacao_historica": 0.91},
    {"id": 2,  "nome": "fone de ouvido bluetooth","preco": 199,  "categoria": "eletronicos",  "pontuacao_historica": 0.85},
    {"id": 3,  "nome": "carregador portatil",     "preco": 89,   "categoria": "eletronicos",  "pontuacao_historica": 0.78},
    {"id": 4,  "nome": "capa protetora",          "preco": 49,   "categoria": "acessorios",   "pontuacao_historica": 0.65},
    {"id": 5,  "nome": "pelicula de vidro",       "preco": 29,   "categoria": "acessorios",   "pontuacao_historica": 0.70},
    {"id": 6,  "nome": "suporte veicular",        "preco": 59,   "categoria": "acessorios",   "pontuacao_historica": 0.68},
    {"id": 7,  "nome": "smartwatch",              "preco": 799,  "categoria": "wearables",    "pontuacao_historica": 0.88},
    {"id": 8,  "nome": "tablet entrada",          "preco": 999,  "categoria": "eletronicos",  "pontuacao_historica": 0.75},
    {"id": 9,  "nome": "notebook ultrafino",      "preco": 4299, "categoria": "computadores", "pontuacao_historica": 0.93},
    {"id": 10, "nome": "mouse sem fio",           "preco": 129,  "categoria": "informatica",  "pontuacao_historica": 0.82},
    {"id": 11, "nome": "teclado mecanico",        "preco": 349,  "categoria": "informatica",  "pontuacao_historica": 0.87},
    {"id": 12, "nome": "webcam hd",               "preco": 249,  "categoria": "informatica",  "pontuacao_historica": 0.79},
    {"id": 13, "nome": "hub usb",                 "preco": 99,   "categoria": "informatica",  "pontuacao_historica": 0.74},
    {"id": 14, "nome": "ssd externo",             "preco": 299,  "categoria": "armazenamento","pontuacao_historica": 0.90},
]

NUM_PRODUTOS = len(CATALOGO_PRODUTOS)
# quantos produtos queremos recomendar no pacote final
TAMANHO_RECOMENDACAO = 5
# orcamento maximo do usuario (simulado)
ORCAMENTO_MAX = 1500.0


#  representacao do individuo 
# cada individuo eh um vetor binario de tamanho NUM_PRODUTOS
# 1 = produto incluido na recomendacao, 0 = nao incluido

def criar_individuo_aleatorio():
    # gera individuo com exatamente TAMANHO_RECOMENDACAO produtos selecionados
    individuo = [0] * NUM_PRODUTOS
    selecionados = random.sample(range(NUM_PRODUTOS), TAMANHO_RECOMENDACAO)
    for idx in selecionados:
        individuo[idx] = 1
    return individuo


def criar_populacao(tamanho):
    return [criar_individuo_aleatorio() for _ in range(tamanho)]


#  funcao de fitness 

def calcular_fitness(individuo, score_fuzzy, orcamento=ORCAMENTO_MAX):
    produtos_selecionados = [CATALOGO_PRODUTOS[i] for i, bit in enumerate(individuo) if bit == 1]

    if not produtos_selecionados:
        return 0.0

    preco_total = sum(p["preco"] for p in produtos_selecionados)
    pontuacao_media = sum(p["pontuacao_historica"] for p in produtos_selecionados) / len(produtos_selecionados)

    # penaliza fortemente se ultrapassar orcamento
    if preco_total > orcamento:
        penalidade_orcamento = 0.3 * (preco_total - orcamento) / orcamento
    else:
        penalidade_orcamento = 0.0

    # bonus por diversidade de categorias (evita recomendar tudo da mesma categoria)
    categorias = set(p["categoria"] for p in produtos_selecionados)
    bonus_diversidade = len(categorias) / TAMANHO_RECOMENDACAO * 0.2

    # fator do score fuzzy normalizado (0 a 1)
    fator_fuzzy = score_fuzzy / 100.0

    # custo-beneficio normalizado (quanto menor o preco, melhor o custo-beneficio relativo)
    custo_beneficio = 1.0 - min(preco_total / (orcamento * 1.5), 1.0)

    # fitness final: combinacao ponderada dos criterios
    fitness = (
        0.35 * pontuacao_media         # qualidade historica dos produtos
        + 0.30 * fator_fuzzy            # score fuzzy da analise de sentimento
        + 0.20 * custo_beneficio        # relacao custo-beneficio
        + 0.15 * bonus_diversidade      # diversidade de categorias
        - penalidade_orcamento          # penalidade por extrapolar orcamento
    )

    return max(0.0, fitness)


#  operadores geneticos 

def selecao_torneio(populacao, fitnesses, k=3):
    # seleciona um individuo por torneio com k participantes
    participantes = random.sample(range(len(populacao)), k)
    melhor = max(participantes, key=lambda i: fitnesses[i])
    return populacao[melhor][:]


def crossover_ponto_unico(pai1, pai2):
    # crossover de ponto unico mantendo o numero de bits ativados
    ponto = random.randint(1, NUM_PRODUTOS - 1)
    filho1 = pai1[:ponto] + pai2[ponto:]
    filho2 = pai2[:ponto] + pai1[ponto:]
    return filho1, filho2


def corrigir_individuo(individuo):
    # garante que o individuo tenha exatamente TAMANHO_RECOMENDACAO produtos
    selecionados = [i for i, b in enumerate(individuo) if b == 1]
    nao_selecionados = [i for i, b in enumerate(individuo) if b == 0]

    while len(selecionados) > TAMANHO_RECOMENDACAO:
        remover = random.choice(selecionados)
        individuo[remover] = 0
        selecionados.remove(remover)
        nao_selecionados.append(remover)

    while len(selecionados) < TAMANHO_RECOMENDACAO:
        adicionar = random.choice(nao_selecionados)
        individuo[adicionar] = 1
        selecionados.append(adicionar)
        nao_selecionados.remove(adicionar)

    return individuo


def mutacao(individuo, taxa_mutacao=0.1):
    # mutacao por troca: troca um produto selecionado por um nao selecionado
    if random.random() < taxa_mutacao:
        selecionados = [i for i, b in enumerate(individuo) if b == 1]
        nao_selecionados = [i for i, b in enumerate(individuo) if b == 0]

        if selecionados and nao_selecionados:
            remover = random.choice(selecionados)
            adicionar = random.choice(nao_selecionados)
            individuo[remover] = 0
            individuo[adicionar] = 1

    return individuo


#  algoritmo genetico principal 

def algoritmo_genetico(score_fuzzy, orcamento=ORCAMENTO_MAX,
                        tamanho_populacao=50, num_geracoes=100,
                        taxa_crossover=0.8, taxa_mutacao=0.15):

    # inicializa populacao
    populacao = criar_populacao(tamanho_populacao)

    historico_melhor = []
    melhor_individuo_global = None
    melhor_fitness_global = -1.0

    for geracao in range(num_geracoes):
        # avalia fitness de todos os individuos
        fitnesses = [calcular_fitness(ind, score_fuzzy, orcamento) for ind in populacao]

        # atualiza melhor global
        melhor_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[melhor_idx] > melhor_fitness_global:
            melhor_fitness_global = fitnesses[melhor_idx]
            melhor_individuo_global = populacao[melhor_idx][:]

        historico_melhor.append(melhor_fitness_global)

        # elitismo: mantém o melhor individuo na proxima geracao
        nova_populacao = [melhor_individuo_global[:]]

        # gera nova populacao por selecao, crossover e mutacao
        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao_torneio(populacao, fitnesses)
            pai2 = selecao_torneio(populacao, fitnesses)

            if random.random() < taxa_crossover:
                filho1, filho2 = crossover_ponto_unico(pai1, pai2)
            else:
                filho1, filho2 = pai1[:], pai2[:]

            filho1 = mutacao(corrigir_individuo(filho1), taxa_mutacao)
            filho2 = mutacao(corrigir_individuo(filho2), taxa_mutacao)

            nova_populacao.append(corrigir_individuo(filho1))
            if len(nova_populacao) < tamanho_populacao:
                nova_populacao.append(corrigir_individuo(filho2))

        populacao = nova_populacao

    # retorna o resultado final
    produtos_recomendados = [
        CATALOGO_PRODUTOS[i]
        for i, bit in enumerate(melhor_individuo_global)
        if bit == 1
    ]

    preco_total = sum(p["preco"] for p in produtos_recomendados)
    categorias = set(p["categoria"] for p in produtos_recomendados)

    return {
        "produtos_recomendados": produtos_recomendados,
        "fitness_final": round(melhor_fitness_global, 4),
        "preco_total": preco_total,
        "dentro_orcamento": preco_total <= orcamento,
        "categorias_cobertas": list(categorias),
        "historico_convergencia": historico_melhor,
        "geracoes_executadas": num_geracoes
    }


if __name__ == "__main__":
    print("camada 3: algoritmo genetico para recomendacao de produtos\n")

    cenarios = [
        (85.0, 1500, "usuario satisfeito com bom orcamento"),
        (20.0, 500,  "usuario insatisfeito com orcamento baixo"),
        (50.0, 1000, "usuario neutro com orcamento medio"),
    ]

    for score, orcamento, descricao in cenarios:
        print(f"cenario: {descricao}")
        print(f"  score fuzzy={score:.1f}, orcamento=R${orcamento}")

        resultado = algoritmo_genetico(score_fuzzy=score, orcamento=orcamento)

        print(f"  fitness final: {resultado['fitness_final']:.4f}")
        print(f"  preco total: R${resultado['preco_total']}")
        print(f"  dentro do orcamento: {resultado['dentro_orcamento']}")
        print(f"  categorias: {resultado['categorias_cobertas']}")
        print(f"  produtos recomendados:")
        for p in resultado["produtos_recomendados"]:
            print(f"    - {p['nome']} (R${p['preco']}, avaliacao historica={p['pontuacao_historica']:.2f})")
        