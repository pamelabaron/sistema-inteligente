import random
import math

SEED = 7  # seed fixa para reprodutibilidade

CATALOGO = [
    # id, nome, preco, categoria, avaliacao_media, specs_pontos
    {"id": 0,  "nome": "Notebook Lenovo IdeaPad 3",     "preco": 2799, "cat": "notebook",    "aval": 0.82, "specs": 0.70},
    {"id": 1,  "nome": "Notebook Dell Inspiron 15",      "preco": 3299, "cat": "notebook",    "aval": 0.88, "specs": 0.80},
    {"id": 2,  "nome": "SSD Kingston 480GB",             "preco": 229,  "cat": "armazenamento","aval": 0.85, "specs": 0.75},
    {"id": 3,  "nome": "SSD Samsung 970 EVO 1TB",        "preco": 649,  "cat": "armazenamento","aval": 0.95, "specs": 0.95},
    {"id": 4,  "nome": "Memória RAM 8GB DDR4 Kingston",  "preco": 189,  "cat": "memoria",     "aval": 0.80, "specs": 0.70},
    {"id": 5,  "nome": "Memória RAM 16GB DDR4 Crucial",  "preco": 319,  "cat": "memoria",     "aval": 0.90, "specs": 0.88},
    {"id": 6,  "nome": "Monitor LG 24\" Full HD",        "preco": 899,  "cat": "monitor",     "aval": 0.84, "specs": 0.78},
    {"id": 7,  "nome": "Monitor Samsung 27\" QHD",       "preco": 1599, "cat": "monitor",     "aval": 0.91, "specs": 0.90},
    {"id": 8,  "nome": "Teclado Mecânico Redragon K552", "preco": 249,  "cat": "periferico",  "aval": 0.87, "specs": 0.80},
    {"id": 9,  "nome": "Mouse Logitech MX Master 3",     "preco": 529,  "cat": "periferico",  "aval": 0.93, "specs": 0.90},
    {"id": 10, "nome": "Webcam Logitech C920",           "preco": 449,  "cat": "periferico",  "aval": 0.89, "specs": 0.85},
    {"id": 11, "nome": "Headset HyperX Cloud II",        "preco": 499,  "cat": "audio",       "aval": 0.91, "specs": 0.88},
    {"id": 12, "nome": "Placa de Vídeo RTX 3060",        "preco": 2199, "cat": "gpu",         "aval": 0.90, "specs": 0.92},
    {"id": 13, "nome": "Processador Ryzen 5 5600X",      "preco": 1099, "cat": "processador", "aval": 0.93, "specs": 0.94},
    {"id": 14, "nome": "Fonte Corsair 650W 80+ Gold",    "preco": 599,  "cat": "fonte",       "aval": 0.88, "specs": 0.86},
]

N = len(CATALOGO)
N_RECOMENDACOES = 4  # quantos produtos por recomendação


# parâmetros do ga 

PARAMS_GA = {
    "tamanho_populacao": 6,      # population size pequena e visível
    "num_geracoes": 80,
    "prob_cruzamento": 0.85,     # probabilidade de cruzamento entre dois pais
    "prob_mutacao": 0.12,        # probabilidade de mutação por indivíduo
    "k_torneio": 3,              # participantes no torneio de seleção
}



# representação e inicialização


def criar_individuo(rng):
    ind = [0] * N
    for i in rng.sample(range(N), N_RECOMENDACOES):
        ind[i] = 1
    return ind


def criar_populacao(rng, tam):
    return [criar_individuo(rng) for _ in range(tam)]


def corrigir(ind, rng):
    #garante exatamente N_RECOMENDACOES bits = 1
    ativos = [i for i, b in enumerate(ind) if b == 1]
    inativos = [i for i, b in enumerate(ind) if b == 0]

    while len(ativos) > N_RECOMENDACOES:
        rem = rng.choice(ativos)
        ind[rem] = 0
        ativos.remove(rem)
        inativos.append(rem)

    while len(ativos) < N_RECOMENDACOES:
        add = rng.choice(inativos)
        ind[add] = 1
        ativos.append(add)
        inativos.remove(add)

    return ind



# função de fitness
# combina: qualidade histórica + specs + score fuzzy + custo-benefício

def fitness(individuo, score_fuzzy, orcamento):
    selecionados = [CATALOGO[i] for i, b in enumerate(individuo) if b == 1]
    if not selecionados:
        return 0.0

    preco_total = sum(p["preco"] for p in selecionados)
    aval_media = sum(p["aval"] for p in selecionados) / len(selecionados)
    specs_media = sum(p["specs"] for p in selecionados) / len(selecionados)

    # diversidade de categorias
    cats = set(p["cat"] for p in selecionados)
    bonus_diversidade = len(cats) / N_RECOMENDACOES

    # penalidade proporcional se ultrapassar orçamento
    if preco_total > orcamento:
        penalidade = 0.4 * (preco_total - orcamento) / orcamento
    else:
        penalidade = 0.0

    # custo-benefício: quanto mais barato em relação ao orçamento, melhor
    cb = 1.0 - min(preco_total / (orcamento * 1.2), 1.0)

    fz = score_fuzzy / 100.0

    # fitness ponderado
    f = (
        0.30 * aval_media
        + 0.25 * specs_media
        + 0.20 * fz
        + 0.15 * cb
        + 0.10 * bonus_diversidade
        - penalidade
    )
    return max(0.0, f)


# operadores genéticos

def selecao_torneio(populacao, fitnesses, k, rng):
    participantes = rng.sample(range(len(populacao)), k)
    melhor = max(participantes, key=lambda i: fitnesses[i])
    return populacao[melhor][:]


def cruzamento_ponto_unico(pai1, pai2, rng):
    ponto = rng.randint(1, N - 1)
    filho1 = pai1[:ponto] + pai2[ponto:]
    filho2 = pai2[:ponto] + pai1[ponto:]
    return filho1, filho2


def mutacao_troca(individuo, rng):
    #troca um gene ativo por um inativo: mantém o número de bits
    ativos = [i for i, b in enumerate(individuo) if b == 1]
    inativos = [i for i, b in enumerate(individuo) if b == 0]
    if ativos and inativos:
        sai = rng.choice(ativos)
        entra = rng.choice(inativos)
        individuo[sai] = 0
        individuo[entra] = 1
    return individuo


# algoritmo genético principal

def algoritmo_genetico(score_fuzzy, orcamento, params=None):
    if params is None:
        params = PARAMS_GA

    rng = random.Random(SEED)

    tam_pop = params["tamanho_populacao"]
    n_ger = params["num_geracoes"]
    p_cruz = params["prob_cruzamento"]
    p_mut = params["prob_mutacao"]
    k_torn = params["k_torneio"]

    populacao = criar_populacao(rng, tam_pop)

    # avalia geração inicial para exibição
    fits_iniciais = [fitness(ind, score_fuzzy, orcamento) for ind in populacao]
    populacao_inicial = [ind[:] for ind in populacao]

    melhor_global = None
    melhor_fit_global = -1.0
    historico = []

    for geracao in range(n_ger):
        fits = [fitness(ind, score_fuzzy, orcamento) for ind in populacao]

        # registra melhor da geração
        idx_melhor = max(range(tam_pop), key=lambda i: fits[i])
        if fits[idx_melhor] > melhor_fit_global:
            melhor_fit_global = fits[idx_melhor]
            melhor_global = populacao[idx_melhor][:]

        historico.append(round(melhor_fit_global, 4))

        # nova geração com elitismo (o melhor sobrevive)
        nova_pop = [melhor_global[:]]

        while len(nova_pop) < tam_pop:
            pai1 = selecao_torneio(populacao, fits, k_torn, rng)
            pai2 = selecao_torneio(populacao, fits, k_torn, rng)

            if rng.random() < p_cruz:
                filho1, filho2 = cruzamento_ponto_unico(pai1, pai2, rng)
            else:
                filho1, filho2 = pai1[:], pai2[:]

            filho1 = corrigir(filho1, rng)
            filho2 = corrigir(filho2, rng)

            if rng.random() < p_mut:
                filho1 = mutacao_troca(filho1, rng)
            if rng.random() < p_mut:
                filho2 = mutacao_troca(filho2, rng)

            nova_pop.append(filho1)
            if len(nova_pop) < tam_pop:
                nova_pop.append(filho2)

        populacao = nova_pop

    produtos_rec = [CATALOGO[i] for i, b in enumerate(melhor_global) if b == 1]
    preco_total = sum(p["preco"] for p in produtos_rec)

    return {
        "produtos": produtos_rec,
        "fitness_final": round(melhor_fit_global, 4),
        "preco_total": preco_total,
        "dentro_orcamento": preco_total <= orcamento,
        "historico_fitness": historico,
        "populacao_inicial": populacao_inicial,
        "fits_iniciais": fits_iniciais,
        "params": params,
    }


if __name__ == "__main__":
    print("camada 3: algoritmo genético\n")
    cenarios = [
        (78.0, 3000, "usuário satisfeito, orçamento médio"),
        (30.0, 1500, "usuário insatisfeito, orçamento baixo"),
        (55.0, 5000, "usuário neutro, orçamento amplo"),
    ]
    for sf, orc, desc in cenarios:
        print(f"cenário: {desc}")
        res = algoritmo_genetico(sf, orc)
        print(f"  fitness: {res['fitness_final']}  |  total: R${res['preco_total']}  |  no orçamento: {res['dentro_orcamento']}")
        for p in res["produtos"]:
            print(f"    - {p['nome']}  R${p['preco']}")
        print()