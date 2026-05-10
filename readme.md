# Sistema Inteligente de Recomendação

Projeto da disciplina de Inteligência Artificial que integra três técnicas de IA em camadas sequenciais para recomendar produtos com base no sentimento do usuário.

---

## o que o sistema faz

O usuário fornece um texto com sua opinião sobre um produto e o valor dele. O sistema analisa esse texto, mede a intensidade do sentimento, cruza com o valor do produto e usa tudo isso para escolher os melhores produtos a recomendar dentro do orçamento disponível.

```
texto do usuario
      |
      v
[camada 1] naive bayes  -->  probabilidade de sentimento (0 a 1)
      |
      v
[camada 2] fuzzy mamdani  -->  score de prioridade (0 a 100)
      |
      v
[camada 3] algoritmo genetico  -->  5 produtos recomendados
```

---

## estrutura de arquivos

```
sistema_inteligente/
    camada1_pln.py      # Naive Bayes com pré-processamento de texto
    camada2_fuzzy.py    # Sistema de inferência fuzzy Mamdani
    camada3_ga.py       # Algoritmo genético para seleção de produtos
    main.py             # Orquestra as três camadas
    readme.md
```

---

## requisitos

Apenas Python 3.8 ou superior. Nenhuma biblioteca externa é necessária.

Para verificar a versão:

```bash
python --version
```

---

## como rodar

 execute:

```bash
python main.py
```

O sistema processa 3 cenários automaticamente e imprime o resultado de cada camada no terminal.
    
Para testar cada camada separadamente:

```bash
python camada1_pln.py
python camada2_fuzzy.py
python camada3_ga.py
```

---

## Camadas

### Camada 1: PLN com Naive Bayes

Classifica o sentimento de um texto em positivo, negativo ou neutro.

O pré-processamento passa por quatro etapas: conversão para minúsculo, remoção de pontuação, tokenização e remoção de stop words em português e inglês. Depois, aplica um stemming rudimentar que corta sufixos comuns como ing, tion, mente e cao.

O modelo usa suavização de Laplace para não quebrar quando encontra uma palavra que não estava no treino. A saída é normalizada via softmax, retornando três probabilidades que somam 1.0.

```
Entrada: "produto horrível, chegou quebrado"
Saída:   positivo=0.04  negativo=0.93  neutro=0.03
```

### Camada 2: Sistema de inferência fuzzy (Mamdani)

Recebe a probabilidade positiva da camada 1 e o valor do produto em reais e retorna um score de 0 a 100.

As variáveis linguísticas usadas são:

| Variável | Conjuntos fuzzy |
|---|---|
| sentimento | negativo, neutro, positivo |
| valor do produto | baixo, medio, alto |
| score de saida | muito baixo, baixo, medio, alto, muito alto |

A base tem 9 regras do tipo: "se sentimento for positivo e valor for alto então score é muito alto". A desfuzzificação usa o método do centroide com resolução de 200 pontos.
```
Entrada: prob_positivo=0.96, valor=R$350
Saída:   score_fuzzy = 91.50 / 100
```

### Camada 3: Algoritmo genético

Seleciona os 5 melhores produtos de um catálogo de 15 itens, respeitando o orçamento do usuário.

Cada indivíduo é um vetor binário de 15 posições, onde o valor 1 indica que aquele produto foi selecionado. A função de fitness pondera quatro critérios:

| criterio | peso |
|---|---|
| avaliacao historica do produto | 35% |
| score fuzzy da camada 2 | 30% |
| custo-beneficio (preco vs orcamento) | 20% |
| diversidade de categorias | 15% |

os operadores geneticos usados sao selecao por torneio com k=3, crossover de ponto unico, mutacao por troca e elitismo. uma funcao de correcao garante que todo individuo tenha exatamente 5 produtos apos o crossover e a mutacao.

```
Parâmetros padrão: 50 indivíduos, 100 gerações
Tempo médio de execução: 0.04s por cenário
```

---

## Exemplo de saída

```
Caso 1: usuário satisfeito com produto caro

  Camada 1: classificação POSITIVO
    prob positivo: 0.9592
    prob negativo: 0.0337

  Camada 2: score fuzzy 91.50 / 100
    nível: muito alto (melhores recomendações disponíveis)

  Camada 3: fitness final 0.7170
    produtos recomendados:
      - carregador portátil    R$89
      - película de vidro      R$29
      - mouse sem fio          R$129
      - hub usb                R$99
      - ssd externo            R$299
    total: R$645 de R$1500 disponíveis
```

---

## como personalizar

**mudar os cenarios de teste** no `main.py`, edite a lista `casos`:

```python
casos = [
    {
        "descricao": "cenário",
        "review": "texto da avaliação",
        "valor_produto": 250.0,
        "orcamento": 1200.0
    },
]
```

**mudar o catalogo de produtos** no `camada3_ga.py`, edite a lista `CATALOGO_PRODUTOS`:

```python
{"id": 0, "nome": "nome do produto", "preço": 199,
 "categoria": "eletrônicos", "pontuacao_historica": 0.80}
```

**ajustar o algoritmo genetico** na chamada dentro do `main.py`:

```python
resultado_ga = algoritmo_genetico(
    score_fuzzy=score_fuzzy,
    orcamento=orcamento_usuario,
    tamanho_populacao=50,   # aumente para busca mais ampla
    num_geracoes=100        # aumente para melhor convergência
)
```

