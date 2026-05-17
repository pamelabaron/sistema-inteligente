# Sistema Inteligente de Recomendação de Eletrônicos

Projeto da disciplina de Inteligência Artificial que integra três técnicas de IA em camadas sequenciais para recomendar produtos eletrônicos de computador com base no comentário do cliente.
---

## Alunos
### Pâmela Baron e Dereck Conink

---

## O que o sistema faz

O cliente digita um comentário sobre um produto eletrônico. O sistema analisa o sentimento do texto, extrai a percepção de preço e qualidade via lógica fuzzy e usa um algoritmo genético para selecionar os melhores produtos a recomendar dentro do orçamento disponível.


---

## Estrutura de arquivos

```
projeto_ia/
    camada1_pln.py    # Naive Bayes com pré-processamento de texto
    camada2_fuzzy.py  # Sistema de inferência fuzzy Mamdani
    camada3_ga.py     # Algoritmo genético para seleção de produtos
    main.py           # Orquestra as três camadas no terminal
    index.html        # Interface web
    README.md
```

---

## Requisitos

 Python 3.8 ou superior. 

Para verificar a versão:

```bash
python --version
```

---

## Como rodar

**No terminal**
```bash
python main.py
```

O sistema processa 4 cenários de teste automaticamente, exibindo o resultado das três camadas para cada um.

**Interface web** : abra o arquivo `index.html` diretamente no navegador.
```bash
start index.html
```

Para testar cada camada separadamente:

```bash
python camada1_pln.py
python camada2_fuzzy.py
python camada3_ga.py
```

---

## Camadas

### Camada 1: PLN com Naive Bayes

Classifica o sentimento de um comentário em positivo, negativo ou neutro, treinado com avaliações de produtos eletrônicos em português e inglês.

O pré-processamento passa por quatro etapas: conversão para minúsculo, remoção de pontuação, tokenização e remoção de palavras irrelevantes em português e inglês. Em seguida, aplica um stemming rudimentar que corta sufixos comuns como `ing`, `tion`, `mente` e `ção`. Negações como "não" e "never" são tratadas com o prefixo `NEG_` no token seguinte, preservando o sentido semântico.

O modelo usa suavização de Laplace para não quebrar com palavras fora do vocabulário de treino. A saída é normalizada via softmax, retornando três probabilidades que somam 1,0.

```
Entrada: "notebook excelente mas muito caro, processador rápido"
Saída:   positivo=0.944  negativo=0.039  neutro=0.017
```

### Camada 2: Sistema de Inferência Fuzzy (Mamdani)

A Camada 2 não recebe apenas a probabilidade do Naive Bayes. Ela extrai diretamente do texto duas percepções independentes usando um léxico calibrado de palavras-chave.

| Variável de entrada | Escala | Exemplos de termos |
|---|---|---|
| Percepção de preço | 0 = barato, 100 = caro | "caro", "salgado", "barato", "em conta" |
| Percepção de qualidade | 0 = ruim, 100 = ótimo | "excelente", "incrível", "péssimo", "lixo" |

Essa abordagem permite capturar comentários como "muito caro mas qualidade excepcional": o preço puxa o score para baixo enquanto a qualidade puxa para cima, e o sistema fuzzy pondera as duas forças corretamente.

As variáveis linguísticas de saída são:

| Conjunto | Faixa aproximada |
|---|---|
| Não recomendar | 0 a 30 |
| Cautela | 15 a 50 |
| Considerar | 35 a 65 |
| Recomendar | 55 a 85 |
| Top escolha | 75 a 100 |

A base contém 10 regras que cobrem todas as combinações de preço e qualidade, incluindo uma regra especial para qualidade excepcional que eleva o score independentemente do preço. A desfuzzificação usa o método do centroide com resolução de 300 pontos.

```
Entrada: "muito caro mas qualidade excepcional"
  Preço percebido:     100/100
  Qualidade percebida: 100/100
Saída: score_fuzzy = 90,1 / 100
```

### Camada 3: Algoritmo Genético

Seleciona os 4 melhores produtos de um catálogo de 15 eletrônicos, respeitando o orçamento do cliente. O algoritmo não é aleatório puro: utiliza semente fixa para reprodutibilidade e elitismo garantido.

**Representação:** cada indivíduo é um vetor binário de 15 posições. O valor 1 indica produto selecionado. Uma função de correção garante exatamente 4 bits ativos após qualquer operação genética.

**Função de fitness:**

| Critério | Peso |
|---|---|
| Avaliação histórica do produto | 30% |
| Especificações técnicas | 25% |
| Score fuzzy da Camada 2 | 20% |
| Custo-benefício (preço versus orçamento) | 15% |
| Diversidade de categorias | 10% |

**Operadores genéticos:**

| Operador | Detalhes |
|---|---|
| Seleção | Torneio com k = 3 participantes |
| Cruzamento | Ponto único, probabilidade padrão de 0,85 |
| Mutação | Troca de gene ativo por inativo, probabilidade padrão de 0,12 |
| Elitismo | O melhor indivíduo sempre sobrevive para a próxima geração |

```
Parâmetros padrão: 6 indivíduos, 80 gerações
Tempo médio de execução: menos de 0,01 s por caso
```

---

## Como personalizar

**Cenários de teste** — no `main.py`, edite a lista `CASOS`:

```python
CASOS = [
    {
        "descricao": "seu cenário",
        "review":    "texto do comentário aqui",
        "orcamento": 2000,
    },
]
```

**Catálogo de produtos** — no `camada3_ga.py`, edite a lista `CATALOGO`:

```python
{"id": 0, "nome": "Nome do Produto", "preco": 499,
 "cat": "periferico", "aval": 0.85, "specs": 0.80}
```

**Parâmetros do algoritmo genético** — no `camada3_ga.py`, edite o dicionário `PARAMS_GA`:

```python
PARAMS_GA = {
    "tamanho_populacao": 6,     # aumente para uma busca mais ampla
    "num_geracoes": 80,         # aumente para melhor convergência
    "prob_cruzamento": 0.85,    # probabilidade de cruzamento
    "prob_mutacao": 0.12,       # probabilidade de mutação
    "k_torneio": 3,             # participantes no torneio
}
```

