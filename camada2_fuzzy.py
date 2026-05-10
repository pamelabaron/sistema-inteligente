import math


# funcoes de pertinencia 

def triangular(x, a, b, c):
    # funcao de pertinencia triangular
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapezoidal(x, a, b, c, d):
    # funcao de pertinencia trapezoidal
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


#  pertinencias para prob_positivo 0 a 1

def sentimento_negativo(prob):
    # alta pertinencia quando prob positivo é baixa - sentimento  negativo
    return trapezoidal(prob, 0.0, 0.0, 0.2, 0.45)


def sentimento_neutro(prob):
    return triangular(prob, 0.3, 0.5, 0.7)


def sentimento_positivo(prob):
    return trapezoidal(prob, 0.55, 0.75, 1.0, 1.0)


# pertinencias para valor_produto 0 a 500 reais

def valor_baixo(v):
    return trapezoidal(v, 0, 0, 50, 150)


def valor_medio(v):
    return triangular(v, 80, 200, 350)


def valor_alto(v):
    return trapezoidal(v, 280, 400, 500, 500)


# pertinencias para score_prioridade 0 a 100
# essa é a variavel de saida do sistema fuzzy

def prioridade_muito_baixa(x):
    return trapezoidal(x, 0, 0, 10, 25)


def prioridade_baixa(x):
    return triangular(x, 10, 25, 45)


def prioridade_media(x):
    return triangular(x, 35, 50, 65)


def prioridade_alta(x):
    return triangular(x, 55, 75, 90)


def prioridade_muito_alta(x):
    return trapezoidal(x, 80, 92, 100, 100)


#  base de regras fuzzy 
# formato: grau_sentimento, grau_valor ->  conjunto_saida, funcao_saida
# regras baseadas em conhecimento de dominio de e-commerce

def avaliar_regras(prob_positivo, valor_produto):
    # calcula graus de pertinencia das entradas
    neg = sentimento_negativo(prob_positivo)
    neu = sentimento_neutro(prob_positivo)
    pos = sentimento_positivo(prob_positivo)

    v_baixo = valor_baixo(valor_produto)
    v_medio = valor_medio(valor_produto)
    v_alto = valor_alto(valor_produto)

    regras = []

    # regra 1 sentimento negativo com valor baixo -> prioridade muito baixa
    regras.append((min(neg, v_baixo), prioridade_muito_baixa))

    # regra 2 sentimento negativo com valor medio -> prioridade baixa
    regras.append((min(neg, v_medio), prioridade_baixa))

    # regra 3 sentimento negativo com valor alto -> prioridade media (cliente pagou caro)
    regras.append((min(neg, v_alto), prioridade_media))

    # regra 4 sentimento neutro com qualquer valor -> prioridade media
    regras.append((min(neu, v_baixo), prioridade_baixa))
    regras.append((min(neu, v_medio), prioridade_media))
    regras.append((min(neu, v_alto), prioridade_media))

    # regra 5 sentimento positivo com valor baixo -> prioridade media
    regras.append((min(pos, v_baixo), prioridade_media))

    # regra 6 sentimento positivo com valor medio -> prioridade alta
    regras.append((min(pos, v_medio), prioridade_alta))

    # regra 7 sentimento positivo com valor alto -> prioridade muito alta
    regras.append((min(pos, v_alto), prioridade_muito_alta))

    return regras


#  desfuzzificacao pelo metodo do centroide

def defuzzificar(regras, resolucao=200):
    # discretiza o universo de saida de 0 a 100
    universo = [i * 100 / resolucao for i in range(resolucao + 1)]

    numerador = 0.0
    denominador = 0.0

    for x in universo:
        # aplica agregacao maxima (mamdani)
        ativacao_max = 0.0
        for (grau_regra, func_saida) in regras:
            # recorte da funçao de saida pelo grau de ativacao da regra
            ativacao = min(grau_regra, func_saida(x))
            ativacao_max = max(ativacao_max, ativacao)

        numerador += x * ativacao_max
        denominador += ativacao_max

    if denominador == 0:
        return 50.0  # valor padrao se nenhuma regra foi ativada

    return numerador / denominador


# interface principal

def calcular_score_fuzzy(prob_positivo, valor_produto):
    # prob_positivo: saida do naive bayes (0.0 a 1.0)
    # valor_produto: valor numerico do produto em reais (0 a 500)

    regras = avaliar_regras(prob_positivo, valor_produto)
    score = defuzzificar(regras)

    # coleta graus de pertinencia para debug
    detalhes = {
        "entradas": {
            "prob_positivo": prob_positivo,
            "valor_produto": valor_produto
        },
        "pertinencias_sentimento": {
            "negativo": round(sentimento_negativo(prob_positivo), 4),
            "neutro":   round(sentimento_neutro(prob_positivo), 4),
            "positivo": round(sentimento_positivo(prob_positivo), 4)
        },
        "pertinencias_valor": {
            "baixo": round(valor_baixo(valor_produto), 4),
            "medio": round(valor_medio(valor_produto), 4),
            "alto":  round(valor_alto(valor_produto), 4)
        },
        "score_fuzzy": round(score, 2)
    }

    return score, detalhes


if __name__ == "__main__":
    print("camada 2: sistema de inferencia fuzzy\n")

    casos = [
        (0.85, 300, "cliente positivo com produto caro"),
        (0.15, 50,  "cliente negativo com produto barato"),
        (0.50, 200, "cliente neutro com produto medio"),
        (0.90, 450, "cliente muito positivo com produto muito caro"),
        (0.10, 400, "cliente muito negativo com produto caro"),
    ]

    for prob, valor, descricao in casos:
        score, detalhes = calcular_score_fuzzy(prob, valor)
        print(f"caso: {descricao}")
        print(f"  prob_positivo={prob:.2f}, valor_produto=R${valor:.0f}")
        print(f"  score fuzzy resultante: {score:.2f}/100")
        print(f"  pertinencias sentimento: {detalhes['pertinencias_sentimento']}")
        