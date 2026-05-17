import re


LEXICO_PRECO = {
    # sinaliza produto caro (valor alto = percepção de caro)
    "caro": +35, "caríssimo": +55, "salgado": +30, "custa muito": +40,
    "overpriced": +45, "expensive": +40, "pricey": +30, "absurdo": +45,
    "cobram demais": +40, "preço alto": +35, "custou muito": +40,
    # sinaliza produto barato (valor negativo = percepção de barato)
    "barato": -35, "econômico": -30, "acessível": -25, "custo baixo": -30,
    "cheap": -30, "affordable": -25, "budget": -25, "em conta": -30,
    "bom preço": -25, "vale o preço": -20, "preço justo": -20,
    "custo benefício": -15, "cost effective": -20,
}

LEXICO_QUALIDADE = {
    # alta qualidade
    "excelente": +50, "incrível": +50, "ótimo": +40, "perfeito": +50,
    "fantástico": +50, "excepcional": +55, "top": +40, "premium": +45,
    "rápido": +30, "potente": +35, "eficiente": +35, "durável": +35,
    "confiável": +30, "preciso": +30, "silencioso": +25, "nítido": +30,
    "amazing": +50, "excellent": +50, "great": +40, "perfect": +50,
    "fast": +30, "powerful": +35, "reliable": +30, "smooth": +30,
    "good": +30, "works": +20, "solid": +30,
    # baixa qualidade
    "péssimo": -55, "horrível": -55, "terrível": -50, "lixo": -60,
    "defeito": -50, "quebrou": -45, "travou": -40, "esquenta": -35,
    "ruim": -45, "fraco": -35, "lento": -35, "barulhento": -30,
    "inutilizável": -55, "decepcionante": -45, "frustrado": -40,
    "terrible": -55, "awful": -55, "broken": -50, "defective": -50,
    "bad": -45, "slow": -35, "noisy": -30, "disappointing": -45,
    "worst": -60, "failed": -50,
}


def extrair_percepcoes(texto):
    # analisa o texto e retorna scores de percepção de preço e qualidade. leva em conta negações: "não é caro" inverte o sinal do termo. retorna valores em escala 0-100.
    
    texto_lower = texto.lower()
    palavras = texto_lower.split()

    negacoes = {"não", "nao", "nem", "nunca", "not", "no", "never", "without"}

    def score_lexico(lexico, palavras):
        score = 0
        for i, palavra in enumerate(palavras):
            negar = i > 0 and palavras[i - 1] in negacoes
            for termo, peso in lexico.items():
                if termo in palavra or termo in " ".join(palavras[max(0, i-1):i+3]):
                    score += (-peso if negar else peso)
                    break
        return score

    raw_preco = score_lexico(LEXICO_PRECO, palavras)
    raw_qualidade = score_lexico(LEXICO_QUALIDADE, palavras)

    # normaliza para escala 0-100
    # preço com score positivo = caro (100), negativo = barato (0), zero = médio (50)
    percepcao_preco = max(0, min(100, 50 + raw_preco))
    # qualidade com score positivo = bom (100), negativo = ruim (0), zero = médio (50)
    percepcao_qualidade = max(0, min(100, 50 + raw_qualidade))

    return percepcao_preco, percepcao_qualidade


# funções de pertinência


def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


# pertinências para percepção de preço (0=barato, 100=caro)
def preco_barato(x):    return trapezoidal(x, 0, 0, 20, 40)
def preco_medio(x):     return triangular(x, 25, 50, 75)
def preco_caro(x):      return trapezoidal(x, 60, 80, 100, 100)

# pertinências para percepção de qualidade (0=ruim, 100=ótimo)
def qual_ruim(x):       return trapezoidal(x, 0, 0, 20, 40)
def qual_media(x):      return triangular(x, 25, 50, 75)
def qual_boa(x):        return trapezoidal(x, 60, 80, 100, 100)

# pertinências para o score de recomendação (saída)
def rec_nao_recomendar(x):  return trapezoidal(x, 0, 0, 15, 30)
def rec_cautela(x):         return triangular(x, 15, 30, 50)
def rec_considerar(x):      return triangular(x, 35, 50, 65)
def rec_recomendar(x):      return triangular(x, 55, 70, 85)
def rec_top_escolha(x):     return trapezoidal(x, 75, 90, 100, 100)


# base de regras fuzzy
def avaliar_regras(perc_preco, perc_qualidade):
    pb = preco_barato(perc_preco)
    pm = preco_medio(perc_preco)
    pc = preco_caro(perc_preco)

    qr = qual_ruim(perc_qualidade)
    qm = qual_media(perc_qualidade)
    qb = qual_boa(perc_qualidade)

    regras = []

    # preço baixo
    regras.append((min(pb, qr), rec_cautela))          # barato e ruim: cautela
    regras.append((min(pb, qm), rec_considerar))       # barato e médio: considerar
    regras.append((min(pb, qb), rec_recomendar))       # barato e bom: recomendar

    # preço médio
    regras.append((min(pm, qr), rec_nao_recomendar))   # médio e ruim: não recomendar
    regras.append((min(pm, qm), rec_considerar))       # médio e médio: considerar
    regras.append((min(pm, qb), rec_recomendar))       # médio e bom: recomendar

    # preço alto 
    regras.append((min(pc, qr), rec_nao_recomendar))   # caro e ruim: não recomendar
    regras.append((min(pc, qm), rec_cautela))          # caro e médio: cautela
    regras.append((min(pc, qb), rec_considerar))       # caro e bom: considerar o custo benefício médio

    # regra extra: qualidade excepcional independe do preço
    qualidade_muito_boa = max(0, (perc_qualidade - 85) / 15)
    regras.append((qualidade_muito_boa * 0.8, rec_top_escolha))

    return regras


def defuzzificar(regras, resolucao=300):
    universo = [i * 100 / resolucao for i in range(resolucao + 1)]
    num = 0.0
    den = 0.0
    for x in universo:
        ativacao = 0.0
        for (grau, func) in regras:
            ativacao = max(ativacao, min(grau, func(x)))
        num += x * ativacao
        den += ativacao
    return num / den if den > 0 else 50.0


def calcular_score_fuzzy(texto=None, perc_preco=None, perc_qualidade=None):
    #aceita texto OU valores diretos de percepção. retorna score (0-100) e dicionário com detalhes.
    
    if texto is not None:
        perc_preco, perc_qualidade = extrair_percepcoes(texto)

    regras = avaliar_regras(perc_preco, perc_qualidade)
    score = defuzzificar(regras)

    detalhes = {
        "percepcao_preco": round(perc_preco, 1),
        "percepcao_qualidade": round(perc_qualidade, 1),
        "pertinencias_preco": {
            "barato": round(preco_barato(perc_preco), 3),
            "medio":  round(preco_medio(perc_preco), 3),
            "caro":   round(preco_caro(perc_preco), 3),
        },
        "pertinencias_qualidade": {
            "ruim":  round(qual_ruim(perc_qualidade), 3),
            "media": round(qual_media(perc_qualidade), 3),
            "boa":   round(qual_boa(perc_qualidade), 3),
        },
        "score_fuzzy": round(score, 2),
    }
    return score, detalhes


if __name__ == "__main__":
    print("camada 2: sistema de inferência fuzzy\n")
    casos = [
        "muito caro mas muito bom, qualidade excepcional",
        "barato e péssimo, quebrou na primeira semana",
        "preço médio e qualidade ok, sem reclamações",
        "caro e horrível, não recomendo de jeito nenhum",
        "barato e excelente, melhor custo benefício",
        "preço justo e produto bom, recomendo",
    ]
    for texto in casos:
        score, d = calcular_score_fuzzy(texto=texto)
        print(f"texto: \"{texto}\"")
        print(f"  preço percebido: {d['percepcao_preco']:.0f}/100  |  qualidade percebida: {d['percepcao_qualidade']:.0f}/100")
        print(f"  score fuzzy: {score:.1f}/100")
        print()