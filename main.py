import time

from camada1_pln import obter_modelo
from camada2_fuzzy import calcular_score_fuzzy
from camada3_ga import algoritmo_genetico, PARAMS_GA

# casos de teste
CASOS = [
    {
        "descricao": "cliente satisfeito com produto caro",
        "review":    "notebook excelente mas muito caro, processador rápido e bateria dura o dia todo",
        "orcamento": 3000,
    },
    {
        "descricao": "cliente insatisfeito com produto barato",
        "review":    "ssd parou de funcionar após um mês, péssimo produto, não recomendo de jeito nenhum",
        "orcamento": 1000,
    },
    {
        "descricao": "cliente neutro com orçamento médio",
        "review":    "monitor ok, nada especial, preço razoável, atende o básico",
        "orcamento": 2000,
    },
    {
        "descricao": "produto caro mas qualidade excepcional",
        "review":    "muito caro mas qualidade excepcional, placa de vídeo incrível, vale cada centavo",
        "orcamento": 5000,
    },
]


def linha(char="─", largura=58):
    print(char * largura)


def cabecalho(titulo):
    print()
    linha("═")
    print(f"  {titulo}")
    linha("═")


def secao(num, titulo):
    print()
    linha()
    print(f"  camada {num}: {titulo}")
    linha()


def executar(caso):
    cabecalho(caso["descricao"].upper())
    print(f"  review: \"{caso['review']}\"")
    print(f"  orçamento: R${caso['orcamento']:,.0f}")

    # camada 1: naive bayes
    secao(1, "percepção e sentimento (naive bayes)")

    modelo = obter_modelo()
    resultado_nb = modelo.classificar(caso["review"])

    classe = resultado_nb["classe"]
    probs  = resultado_nb["probabilidades"]
    tokens = resultado_nb["tokens"]

    print(f"\n  classificação: {classe.upper()}")
    print()
    for rotulo in ["positivo", "negativo", "neutro"]:
        pct = probs[rotulo] * 100
        barra = "█" * int(pct / 4)
        print(f"  {rotulo:<10} {barra:<25} {pct:5.1f}%")

    print(f"\n  tokens usados: {', '.join(tokens) if tokens else '(nenhum após preprocessamento)'}")

    # camada 2: fuzzy
    secao(2, "inferência fuzzy (mamdani)")

    score_fuzzy, detalhes = calcular_score_fuzzy(texto=caso["review"])

    pp = detalhes["percepcao_preco"]
    pq = detalhes["percepcao_qualidade"]
    pert_p = detalhes["pertinencias_preco"]
    pert_q = detalhes["pertinencias_qualidade"]

    print(f"\n  percepção de preço:     {pp:.0f}/100  (0=barato, 100=caro)")
    print(f"  percepção de qualidade: {pq:.0f}/100  (0=ruim, 100=ótimo)")

    print(f"\n  pertinências de preço:")
    for k, v in pert_p.items():
        barra = "█" * int(v * 20)
        print(f"    {k:<8} {barra:<20} {v:.3f}")

    print(f"\n  pertinências de qualidade:")
    for k, v in pert_q.items():
        barra = "█" * int(v * 20)
        print(f"    {k:<8} {barra:<20} {v:.3f}")

    print(f"\n  score fuzzy resultante: {score_fuzzy:.2f} / 100")

    # interpreta o score
    if score_fuzzy >= 80:
        nivel = "top escolha"
    elif score_fuzzy >= 65:
        nivel = "recomendar"
    elif score_fuzzy >= 45:
        nivel = "considerar"
    elif score_fuzzy >= 25:
        nivel = "cautela"
    else:
        nivel = "não recomendar"
    print(f"  nível de recomendação:  {nivel}")

    # camada 3: algoritmo genético
    secao(3, "recomendação por algoritmo genético")

    params = dict(PARAMS_GA)

    print(f"\n  parâmetros do ga:")
    print(f"    população:          {params['tamanho_populacao']} indivíduos")
    print(f"    gerações:           {params['num_geracoes']}")
    print(f"    prob. cruzamento:   {params['prob_cruzamento']}")
    print(f"    prob. mutação:      {params['prob_mutacao']}")
    print(f"    torneio (k):        {params['k_torneio']}")
    print(f"    produtos por lista: {N_RECOMENDACOES}")

    print(f"\n  executando...")
    inicio = time.time()
    resultado_ga = algoritmo_genetico(score_fuzzy, caso["orcamento"], params)
    duracao = time.time() - inicio

    print(f"  concluído em {duracao:.3f}s")

    # população inicial
    print(f"\n  população inicial (geração 0):")
    print(f"  {'#':<4} {'produtos selecionados':<38} {'fitness'}")
    linha("-", 58)
    fits = resultado_ga["fits_iniciais"]
    melhor_idx = fits.index(max(fits))
    for i, (ind, fit) in enumerate(zip(resultado_ga["populacao_inicial"], fits)):
        from camada3_ga import CATALOGO
        nomes = [CATALOGO[j]["nome"].split()[0] for j, b in enumerate(ind) if b == 1]
        marcador = " ◀ melhor" if i == melhor_idx else ""
        print(f"  {i+1:<4} {', '.join(nomes):<38} {fit:.4f}{marcador}")

    # convergência resumida
    hist = resultado_ga["historico_fitness"]
    print(f"\n  convergência (fitness a cada 20 gerações):")
    for g in range(0, len(hist), 20):
        barra = "█" * int(hist[g] * 30)
        print(f"    gen {g:>3}: {barra:<30} {hist[g]:.4f}")
    print(f"    gen {len(hist)-1:>3}: {'█' * int(hist[-1]*30):<30} {hist[-1]:.4f}")

    # resultado final
    prods = resultado_ga["produtos"]
    preco_total = resultado_ga["preco_total"]
    ok = resultado_ga["dentro_orcamento"]

    print(f"\n  produtos recomendados:")
    linha("-", 58)
    for i, p in enumerate(prods, 1):
        print(f"  {i}. {p['nome']}")
        print(f"     categoria: {p['cat']}  |  avaliação: {p['aval']*100:.0f}%  |  R${p['preco']:,}")
    linha("-", 58)
    status = "✓ dentro do orçamento" if ok else "✗ acima do orçamento"
    print(f"  total: R${preco_total:,}  |  {status}")
    print(f"  fitness final: {resultado_ga['fitness_final']:.4f}")


def main():
    print()
    print("  sistema inteligente de recomendação de eletrônicos")
    print("  nlp (naive bayes) + fuzzy mamdani + algoritmo genético")
    print("  inteligência artificial — n2")

    for i, caso in enumerate(CASOS):
        executar(caso)
        if i < len(CASOS) - 1:
            print()
            input("  pressione enter para o próximo caso...")

    print()
    linha("═")
    print("  fim da execução")
    linha("═")
    print()


# importa N_RECOMENDACOES para usar no main
from camada3_ga import N_RECOMENDACOES

if __name__ == "__main__":
    main()