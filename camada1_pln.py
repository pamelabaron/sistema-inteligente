import re
import math
from collections import defaultdict

DADOS_TREINO = [
    # positivo
    ("notebook excelente, processador rápido e bateria dura o dia todo", "positivo"),
    ("o ssd transformou meu computador, carregamento instantâneo", "positivo"),
    ("placa de vídeo incrível para jogos, temperatura ótima", "positivo"),
    ("teclado mecânico com ótima resposta tátil, digitação precisa", "positivo"),
    ("monitor com cores vivas e sem ghosting, muito satisfeito", "positivo"),
    ("mouse ergonômico perfeito, sem dor no pulso depois de horas", "positivo"),
    ("headset com som cristalino e microfone sem ruído", "positivo"),
    ("webcam com imagem nítida mesmo com pouca luz", "positivo"),
    ("memória ram chegou funcionando, instalação fácil", "positivo"),
    ("processador potente, roda tudo sem engasgar", "positivo"),
    ("placa mãe compatível com todos os componentes, funcionou de primeira", "positivo"),
    ("fonte silenciosa e estável, sem variação de voltagem", "positivo"),
    ("gabinete com ótimo fluxo de ar, componentes frescos", "positivo"),
    ("hub usb funciona com todos os dispositivos sem travamento", "positivo"),
    ("cabo hdmi transmite 4k sem perda, qualidade excelente", "positivo"),
    ("great laptop, fast processor and all day battery", "positivo"),
    ("amazing ssd upgrade, boots in seconds now", "positivo"),
    ("keyboard feels premium, very responsive keys", "positivo"),
    ("monitor colors are vivid and accurate, no dead pixels", "positivo"),
    ("graphics card runs cool and quiet, excellent performance", "positivo"),

    # negativo
    ("notebook esquenta demais e trava com programas simples", "negativo"),
    ("ssd parou de funcionar após dois meses, horrível", "negativo"),
    ("placa de vídeo com defeito, artefatos na tela desde o primeiro dia", "negativo"),
    ("teclado com teclas que grudam, péssima qualidade", "negativo"),
    ("monitor com tela manchada e brilho irregular, decepcionante", "negativo"),
    ("mouse com clique duplo involuntário, inutilizável", "negativo"),
    ("headset quebrou na dobradiça em uma semana de uso", "negativo"),
    ("webcam desfocada e com atraso, péssima para reuniões", "negativo"),
    ("memória ram incompatível com a placa mãe, não funcionou", "negativo"),
    ("processador chegou amassado, claramente usado", "negativo"),
    ("placa mãe não reconhece o disco, suporte não ajudou", "negativo"),
    ("fonte com ruído alto e desligou meu pc do nada", "negativo"),
    ("gabinete mal acabado, cortei o dedo na chapa", "negativo"),
    ("hub usb esquenta muito e desconecta dispositivos", "negativo"),
    ("cabo hdmi perde sinal o tempo todo, lixo", "negativo"),
    ("laptop overheats and crashes constantly, worst purchase", "negativo"),
    ("ssd failed after one month, do not buy", "negativo"),
    ("keyboard keys are sticky and unresponsive, terrible", "negativo"),
    ("monitor arrived with dead pixels, very disappointed", "negativo"),
    ("gpu artifacts from day one, defective product", "negativo"),

    # neutro
    ("notebook ok para tarefas básicas, nada especial", "neutro"),
    ("ssd razoável, nem rápido nem lento", "neutro"),
    ("teclado comum, funciona mas sem destaque", "neutro"),
    ("monitor aceitável para o preço, nada impressionante", "neutro"),
    ("mouse simples, atende o básico sem mais", "neutro"),
    ("headset mediano, som ok para o valor", "neutro"),
    ("webcam comum, qualidade padrão", "neutro"),
    ("memória funcionando normalmente, instalação tranquila", "neutro"),
    ("processador entregou o que prometeu, sem surpresas", "neutro"),
    ("produto chegou no prazo, dentro do esperado", "neutro"),
    ("laptop is decent for basic tasks, nothing special", "neutro"),
    ("average keyboard, gets the job done", "neutro"),
    ("monitor is okay, colors are acceptable", "neutro"),
]

# stop words em português e inglês
STOP_WORDS = {
    "a", "o", "e", "de", "do", "da", "em", "um", "uma", "para", "com",
    "não", "no", "na", "os", "as", "dos", "das", "ao", "mas", "mais",
    "por", "se", "que", "foi", "ser", "ele", "ela", "nos", "após",
    "the", "is", "it", "i", "and", "to", "of", "this", "very", "so",
    "my", "on", "in", "at", "be", "an", "or", "by", "no", "was",
    "are", "but", "have", "from", "they", "we", "me", "do", "did",
    "for", "all", "day", "one", "after",
}

# prefixos de negação que invertem o sentido da palavra seguinte
NEGACOES = {"não", "nao", "nem", "nunca", "jamais", "without", "not", "no", "never"}


def preprocessar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Záàâãéèêíïóôõöúüçñ\s]', '', texto)
    tokens_brutos = texto.split()

    tokens = []
    negar_proximo = False
    for token in tokens_brutos:
        if token in NEGACOES:
            negar_proximo = True
            continue
        if token in STOP_WORDS or len(token) < 3:
            negar_proximo = False
            continue
        # aplica prefixo NEG_ para capturar negações semânticas
        if negar_proximo:
            tokens.append("NEG_" + token)
            negar_proximo = False
        else:
            tokens.append(token)

    return tokens


def stemming(tokens):
    sufixos = [
        'mente', 'ando', 'endo', 'ação', 'cao', 'dade', 'ismo',
        'ing', 'tion', 'ness', 'ful', 'less', 'able', 'ible', 'ly', 'ed',
    ]
    resultado = []
    for token in tokens:
        prefixo = ""
        raiz = token
        if token.startswith("NEG_"):
            prefixo = "NEG_"
            raiz = token[4:]
        for suf in sufixos:
            if raiz.endswith(suf) and len(raiz) - len(suf) > 3:
                raiz = raiz[: len(raiz) - len(suf)]
                break
        resultado.append(prefixo + raiz)
    return resultado


class NaiveBayesSentimento:
    def __init__(self):
        self.cont_palavras = defaultdict(lambda: defaultdict(int))
        self.cont_docs = defaultdict(int)
        self.vocabulario = set()
        self.prob_priori = {}
        self.total_docs = 0

    def treinar(self, dados):
        for texto, rotulo in dados:
            tokens = stemming(preprocessar(texto))
            self.cont_docs[rotulo] += 1
            self.total_docs += 1
            for token in tokens:
                self.cont_palavras[rotulo][token] += 1
                self.vocabulario.add(token)
        for classe in self.cont_docs:
            self.prob_priori[classe] = self.cont_docs[classe] / self.total_docs

    def _log_prob(self, tokens, classe):
        total = sum(self.cont_palavras[classe].values())
        vocab = len(self.vocabulario)
        lp = math.log(self.prob_priori[classe])
        for token in tokens:
            lp += math.log((self.cont_palavras[classe][token] + 1) / (total + vocab))
        return lp

    def classificar(self, texto):
        tokens = stemming(preprocessar(texto))
        scores = {c: self._log_prob(tokens, c) for c in self.cont_docs}
        # softmax para converter em probabilidades
        mx = max(scores.values())
        exp_s = {c: math.exp(s - mx) for c, s in scores.items()}
        soma = sum(exp_s.values())
        probs = {c: v / soma for c, v in exp_s.items()}
        classe = max(probs, key=probs.get)
        return {
            "classe": classe,
            "probabilidades": probs,
            "prob_positivo": probs["positivo"],
            "tokens": tokens,
        }


# instância global do modelo já treinado
_modelo = None

def obter_modelo():
    global _modelo
    if _modelo is None:
        _modelo = NaiveBayesSentimento()
        _modelo.treinar(DADOS_TREINO)
    return _modelo


if __name__ == "__main__":
    modelo = obter_modelo()
    testes = [
        "notebook excelente mas muito caro para o que oferece",
        "ssd parou de funcionar, péssimo produto",
        "monitor ok, nada especial",
        "placa de vídeo incrível, valeu cada centavo",
        "teclado terrível, teclas grudam sempre",
    ]
    print("camada 1: classificador de sentimento\n")
    for t in testes:
        r = modelo.classificar(t)
        print(f"texto: {t}")
        print(f"  classe: {r['classe']}  |  prob_positivo: {r['prob_positivo']:.4f}")
        print()