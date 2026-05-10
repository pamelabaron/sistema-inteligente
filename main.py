import sys
import time
import random

# importa as tres camadas do sistema
from camada1_pln import criar_modelo
from camada2_fuzzy import calcular_score_fuzzy
from camada3_ga import algoritmo_genetico, CATALOGO_PRODUTOS

random.seed(42)


def separador(titulo=""):
    largura = 60
    if titulo:
        espaco = (largura - len(titulo) - 2) // 2
        print("\n" + "=" * espaco + f" {titulo} " + "=" * espaco)
    else:
        print("\n" + "=" * largura)


def executar_sistema(review_usuario, valor_produto, orcamento_usuario):
    
   # executa o pipeline completo das tres camadas para um usuario

   # parametros:
       # review_usuario: texto da avaliacao ou comentario do usuario
       # valor_produto:  valor numerico do produto avaliado (em reais)
       # orcamento_usuario: orcamento disponivel para recomendacoes
    

    separador("camada 1: analise de sentimento")
    print(f"\nreview recebido: \"{review_usuario}\"")
    print(f"valor do produto avaliado: R${valor_produto:.2f}")

    # camada 1: naive bayes classifica o sentimento
    modelo_nb = criar_modelo()
    resultado_nb = modelo_nb.classificar(review_usuario)

    prob_positivo = resultado_nb["probabilidades"]["positivo"]
    prob_negativo = resultado_nb["probabilidades"]["negativo"]
    prob_neutro   = resultado_nb["probabilidades"]["neutro"]

    print(f"\nclassificacao: {resultado_nb['classe'].upper()}")
    print(f"probabilidade positivo: {prob_positivo:.4f}")
    print(f"probabilidade negativo: {prob_negativo:.4f}")
    print(f"probabilidade neutro:   {prob_neutro:.4f}")
    print(f"tokens utilizados: {resultado_nb['tokens_usados']}")

    separador("camada 2: inferencia fuzzy")
    print(f"\nentradas do controlador fuzzy:")
    print(f"  prob_positivo (saida do naive bayes): {prob_positivo:.4f}")
    print(f"  valor do produto: R${valor_produto:.2f}")

    # camada 2: sistema fuzzy calcula o score de prioridade
    score_fuzzy, detalhes_fuzzy = calcular_score_fuzzy(prob_positivo, valor_produto)

    pert = detalhes_fuzzy["pertinencias_sentimento"]
    pert_v = detalhes_fuzzy["pertinencias_valor"]

    print(f"\npertinencias de sentimento:")
    print(f"  negativo: {pert['negativo']:.4f}")
    print(f"  neutro:   {pert['neutro']:.4f}")
    print(f"  positivo: {pert['positivo']:.4f}")

    print(f"\npertinencias de valor:")
    print(f"  baixo: {pert_v['baixo']:.4f}")
    print(f"  medio: {pert_v['medio']:.4f}")
    print(f"  alto:  {pert_v['alto']:.4f}")

    print(f"\nscore fuzzy resultante: {score_fuzzy:.2f} / 100")

    # interpreta o score fuzzy de forma qualitativa
    if score_fuzzy < 25:
        nivel = "muito baixo (baixa prioridade de recomendacao)"
    elif score_fuzzy < 45:
        nivel = "baixo (recomendacoes economicas)"
    elif score_fuzzy < 60:
        nivel = "medio (recomendacoes balanceadas)"
    elif score_fuzzy < 80:
        nivel = "alto (recomendacoes de qualidade)"
    else:
        nivel = "muito alto (melhores recomendacoes disponíveis)"

    print(f"nivel de prioridade: {nivel}")

    separador("camada 3: algoritmo genetico")
    print(f"\norcamento do usuario: R${orcamento_usuario:.2f}")
    print(f"score fuzzy para o ga: {score_fuzzy:.2f}")
    print("\nexecutando algoritmo genetico (100 geracoes, populacao de 50 individuos)...")

    inicio = time.time()

    # camada 3: ga seleciona a melhor combinacao de produtos
    resultado_ga = algoritmo_genetico(
        score_fuzzy=score_fuzzy,
        orcamento=orcamento_usuario,
        tamanho_populacao=50,
        num_geracoes=100
    )

    tempo = time.time() - inicio

    print(f"concluido em {tempo:.2f}s")
    print(f"\nfitness da solucao final: {resultado_ga['fitness_final']:.4f}")
    print(f"preco total da recomendacao: R${resultado_ga['preco_total']:.2f}")
    print(f"dentro do orcamento: {'sim' if resultado_ga['dentro_orcamento'] else 'NAO'}")
    print(f"categorias cobertas: {', '.join(resultado_ga['categorias_cobertas'])}")

    separador("recomendacao final")
    print("\nprodutos recomendados para o usuario:")
    for i, produto in enumerate(resultado_ga["produtos_recomendados"], 1):
        print(f"\n  {i}. {produto['nome']}")
        print(f"     preco: R${produto['preco']:.2f}")
        print(f"     categoria: {produto['categoria']}")
        print(f"     avaliacao historica: {produto['pontuacao_historica']:.2f}")

    print(f"\ntotal a pagar: R${resultado_ga['preco_total']:.2f} de R${orcamento_usuario:.2f} disponíveis")

    separador()

    return {
        "sentimento": resultado_nb,
        "score_fuzzy": score_fuzzy,
        "recomendacoes": resultado_ga
    }


def main():
    print("\n" + "#" * 60)
    print("#        sistema inteligente de recomendacao            #")
    print("#   nlp + fuzzy inference + genetic algorithm           #")
    print("#" * 60)

    # casos de teste representando diferentes perfis de usuario
    casos = [
        {
            "descricao": "usuario satisfeito com produto caro",
            "review": "this product is absolutely fantastic, i love it so much, best purchase ever",
            "valor_produto": 350.0,
            "orcamento": 1500.0
        },
        {
            "descricao": "usuario insatisfeito com produto barato",
            "review": "terrible product, completely broken, worst purchase ever, very disappointed",
            "valor_produto": 45.0,
            "orcamento": 600.0
        },
        {
            "descricao": "usuario neutro com produto de valor medio",
            "review": "produto razoavel, atende o basico mas poderia ser melhor",
            "valor_produto": 180.0,
            "orcamento": 1000.0
        },
    ]

    for i, caso in enumerate(casos, 1):
        print(f"\n\n{'#' * 60}")
        print(f"#  caso {i}: {caso['descricao']}")
        print(f"{'#' * 60}")

        executar_sistema(
            review_usuario=caso["review"],
            valor_produto=caso["valor_produto"],
            orcamento_usuario=caso["orcamento"]
        )

        if i < len(casos):
            print("\n[processando proximo caso...]\n")


if __name__ == "__main__":
    main()