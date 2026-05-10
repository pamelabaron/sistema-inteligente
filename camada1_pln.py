import re
import math
import json
import random
from collections import defaultdict

# dados de treino simulando reviews de produtos em portugues e ingles
# cada entrada tem o texto e o rotulo positivo, negativo, neutro
dados_treino = [
    ("this product is amazing, i love it so much", "positivo"),
    ("absolutely wonderful experience, highly recommend", "positivo"),
    ("great quality, fast delivery, very happy", "positivo"),
    ("excellent product, works perfectly fine", "positivo"),
    ("fantastic item, exceeded my expectations", "positivo"),
    ("really good value for money, satisfied", "positivo"),
    ("perfect condition, arrived quickly", "positivo"),
    ("outstanding quality, will buy again", "positivo"),
    ("superb product, very pleased with purchase", "positivo"),
    ("best purchase ever, totally worth it", "positivo"),
    ("produto otimo, muito satisfeito com a compra", "positivo"),
    ("excelente qualidade, recomendo muito", "positivo"),
    ("chegou rapido, produto perfeito", "positivo"),
    ("adorei o produto, superou expectativas", "positivo"),
    ("muito bom, valeu cada centavo", "positivo"),
    ("terrible product, completely broken on arrival", "negativo"),
    ("worst purchase ever, total waste of money", "negativo"),
    ("very disappointed, does not work at all", "negativo"),
    ("horrible quality, fell apart after one day", "negativo"),
    ("awful experience, never buying again", "negativo"),
    ("broken item, terrible customer service", "negativo"),
    ("disgusting quality, complete garbage", "negativo"),
    ("frustrated with this product, very poor", "negativo"),
    ("do not buy this, absolute rubbish", "negativo"),
    ("terrible experience, extremely dissatisfied", "negativo"),
    ("produto pessimo, chegou quebrado", "negativo"),
    ("muito ruim, nao funciona direito", "negativo"),
    ("decepcionante, nao recomendo de jeito nenhum", "negativo"),
    ("horrivel qualidade, perdeu dinheiro", "negativo"),
    ("produto defeituoso, muito frustrado", "negativo"),
    ("it is okay, nothing special about it", "neutro"),
    ("average product, meets basic requirements", "neutro"),
    ("decent quality, does what it says", "neutro"),
    ("not bad not great, just acceptable", "neutro"),
    ("mediocre experience, could be better", "neutro"),
    ("product is fine, no complaints or praise", "neutro"),
    ("ordinary item, works as expected", "neutro"),
    ("neither good nor bad, just average", "neutro"),
    ("produto ok, nada de especial", "neutro"),
    ("razoavel, funciona mas poderia ser melhor", "neutro"),
    ("mais ou menos, atende o basico", "neutro"),
    ("produto comum, sem reclamacoes", "neutro"),
]

# lista de stop words em portugues e ingles
stop_words = {
    "a", "o", "e", "de", "do", "da", "em", "um", "uma", "para", "com", "nao",
    "the", "is", "it", "i", "and", "to", "of", "this", "not", "very", "so",
    "my", "on", "in", "at", "be", "as", "an", "or", "by", "no", "its", "was",
    "are", "but", "have", "from", "they", "we", "me", "he", "she", "do", "did",
    "that", "what", "this", "which", "who", "had", "has", "been", "would",
    "com", "por", "que", "se", "na", "no", "os", "as", "dos", "das", "ao",
    "mais", "mas", "foi", "ser", "uma", "ele", "ela", "nos", "em", "um"
}


def preprocessar(texto):
    # converte para minusculo
    texto = texto.lower()
    # remove pontuacao e caracteres especiais
    texto = re.sub(r'[^a-zA-Záàâãéèêíïóôõöúüçñ\s]', '', texto)
    # tokeniza por espaços
    tokens = texto.split()
    # remove stop words e tokens muito curtos
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens


def aplicar_stemming_simples(tokens):
    # stemming rudimentar para portugues e ingles
    sufixos_pt = ['mente', 'ando', 'endo', 'ando', 'ando', 'ando', 'ção', 'cao', 'dade', 'ismo', 'ista']
    sufixos_en = ['ing', 'tion', 'ness', 'ful', 'less', 'able', 'ible', 'ly', 'ed', 'er', 'est']
    resultado = []
    for token in tokens:
        stem = token
        for sufixo in sufixos_pt + sufixos_en:
            if token.endswith(sufixo) and len(token) - len(sufixo) > 3:
                stem = token[: len(token) - len(sufixo)]
                break
        resultado.append(stem)
    return resultado


class NaiveBayesSentimento:
    def __init__(self):
        # contagem de palavras por classe
        self.contagem_palavras = defaultdict(lambda: defaultdict(int))
        # contagem de documentos por classe
        self.contagem_docs = defaultdict(int)
        # vocabulario total
        self.vocabulario = set()
        # probabilidades a priori de cada classe
        self.prob_classe = {}
        self.total_docs = 0

    def treinar(self, dados):
        for texto, rotulo in dados:
            tokens = aplicar_stemming_simples(preprocessar(texto))
            self.contagem_docs[rotulo] += 1
            self.total_docs += 1
            for token in tokens:
                self.contagem_palavras[rotulo][token] += 1
                self.vocabulario.add(token)

        # calcula probabilidade a priori de cada classe
        for classe in self.contagem_docs:
            self.prob_classe[classe] = self.contagem_docs[classe] / self.total_docs

    def calcular_log_prob(self, tokens, classe):
        total_palavras_classe = sum(self.contagem_palavras[classe].values())
        tamanho_vocab = len(self.vocabulario)

        # log da probabilidade a priori
        log_prob = math.log(self.prob_classe[classe])

        # adiciona log da probabilidade de cada token com suavizacao de laplace
        for token in tokens:
            contagem = self.contagem_palavras[classe][token]
            # suavizacao laplace: adiciona 1 ao numerador e tamanho do vocab ao denominador
            prob_token = (contagem + 1) / (total_palavras_classe + tamanho_vocab)
            log_prob += math.log(prob_token)

        return log_prob

    def classificar(self, texto):
        tokens = aplicar_stemming_simples(preprocessar(texto))

        scores = {}
        for classe in self.contagem_docs:
            scores[classe] = self.calcular_log_prob(tokens, classe)

        # converte log probs para probabilidades normalizadas usando softmax
        max_score = max(scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
        soma_exp = sum(exp_scores.values())

        probabilidades = {c: v / soma_exp for c, v in exp_scores.items()}

        # classe com maior probabilidade
        classe_predita = max(probabilidades, key=probabilidades.get)

        return {
            "classe": classe_predita,
            "probabilidades": probabilidades,
            "tokens_usados": tokens
        }


def criar_modelo():
    modelo = NaiveBayesSentimento()
    modelo.treinar(dados_treino)
    return modelo


if __name__ == "__main__":
    modelo = criar_modelo()

    # testa com alguns exemplos
    testes = [
        "this product is absolutely fantastic",
        "produto horrivel, muito decepcionado",
        "produto razoavel, atende o basico",
        "i love this amazing item so much",
        "terrible quality, broken immediately",
    ]

    print("camada 1: analise de sentimento com naive bayes\n")
    for texto in testes:
        resultado = modelo.classificar(texto)
        print(f"texto: {texto}")
        print(f"  classificacao: {resultado['classe']}")
        print(f"  prob positivo: {resultado['probabilidades']['positivo']:.4f}")
        print(f"  prob negativo: {resultado['probabilidades']['negativo']:.4f}")
        print(f"  prob neutro:   {resultado['probabilidades']['neutro']:.4f}")
        