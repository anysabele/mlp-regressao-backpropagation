# MLP em regressão: experimentos controlados e backpropagation do zero

Dois projetos sobre **redes MLP aplicadas a regressão**

| Projeto | Foco | Notebook |
|---|---|---|
| **2: Comportamento das MLPs** | Como capacidade, ativação, ruído e escala dos dados afetam o aprendizado | `Projeto_2_MLP_experimentos.ipynb` |
| **3: MLP e Backpropagation** | Forward e backward escritos à mão em NumPy, validados contra o autograd do PyTorch | `Projeto_3_MLP_Backpropagation.ipynb` |

## Dados

- **Seno sintético** (`y = sin(x)`, `x ∈ [−2π, 2π]`, 240 amostras), com ou sem ruído gaussiano. Usado no Projeto 2.
- **Concrete Compressive Strength** ([UCI](https://doi.org/10.24432/C5PK67), Yeh, 1998, CC BY 4.0): 1.030 amostras, 8 atributos e 1 alvo contínuo (resistência à compressão em MPa). Usado nos Projetos 2 e 3.

Em todos os experimentos, a divisão é **70% treino / 15% validação / 15% teste**, e o `StandardScaler` é ajustado só no treino. O teste não participa de nenhuma escolha (hiperparâmetros ou época de parada) e é usado apenas na avaliação final.

---

## Projeto 2: Comportamento das MLPs

**Baseline:** 1 camada oculta, 8 neurônios, `tanh`, Adam (lr = 0,01), early stopping manual por validação (paciência 30, máximo 2.000 épocas) e 3 sementes por configuração. O treino é feito com `partial_fit` época a época, guardando as curvas de treino e validação.

**Variáveis estudadas (uma por vez):**

| Código | Fator | Comparação | Dados |
|---|---|---|---|
| V1 | Capacidade | 2 × 8 × 32 neurônios | seno |
| V2 | Ativação | sigmoid × tanh × ReLU | seno |
| V3 / V3b | Generalização e ruído | sem ruído × ruído N(0; 0,1); e 8 × 32 neurônios sob ruído | seno |
| V4 | Escala | atributos brutos × padronizados | Concrete |
| EXT-A | Capacidade | 2 × 8 × 32 neurônios | Concrete |
| EXT-B | Taxa de aprendizado | 0,1 × 0,01 × 0,001 | seno |

**Resultados** (RMSE na validação, média de 3 sementes):

| Experimento | Resultado |
|---|---|
| V1: capacidade (seno) | 2 neurônios: 0,551 · 8: 0,055 · **32: 0,0135** |
| V2: ativação (seno) | sigmoid: 0,705 · ReLU: 0,229 · **tanh: 0,055** |
| V3b: sob ruído, erro contra a função verdadeira | 8 neurônios: 0,073 · **32 neurônios: 0,051** |
| V4: escala (Concrete) | bruto: 16,76 MPa · **padronizado: 6,90 MPa** |
| EXT-A: capacidade (Concrete) | 2: 7,24 · 8: 6,90 · **32: 5,60 MPa** |
| EXT-B: taxa de aprendizado (seno) | 0,1: 0,068 · **0,01: 0,055** · 0,001: 0,278 (não convergiu em 2.000 épocas) |

**Principais observações**
- Padronizar os atributos reduziu o erro no Concrete de 16,8 para 6,9 MPa, o maior efeito entre todos os fatores testados.
- Com 2 neurônios a rede não aproxima o seno (R² ≈ 0,30 na validação). Com 32, o ajuste é praticamente perfeito.
- No seno, a `tanh` foi a melhor ativação. A `sigmoid` ficou bem atrás, com a mesma arquitetura e o mesmo orçamento de épocas.
- Sob ruído, 32 neurônios aproximaram a função verdadeira melhor que 8, sem sinal de que mais capacidade tenha overfitado mais nessa configuração.

---

## Projeto 3: MLP e Backpropagation

Rede MLP para o Concrete com **forward e backward implementados à mão em NumPy**, mostrando as fórmulas (δ, ∂L/∂W, ∂L/∂b) literalmente no código, versão vetorizada para mini-lotes:

```
Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾ @ W⁽ˡ⁾ + b⁽ˡ⁾
dZ⁽ˡ⁻¹⁾ = (dZ⁽ˡ⁾ @ W⁽ˡ⁾ᵀ) ⊙ g'(Z⁽ˡ⁻¹⁾)
dW⁽ˡ⁾ = A⁽ˡ⁻¹⁾ᵀ @ dZ⁽ˡ⁾
```

**Decisões de projeto:** saída linear de 1 neurônio (o alvo é contínuo), perda MSE, métricas MAE, RMSE e R², e gradiente descendente puro com mini-lote de 32 (sem Adam nem momentum).

**Baseline:** 1 camada oculta de 16 neurônios, ReLU, inicialização He, η = 0,01.

**Validação da implementação:**
- O notebook traz um **exemplo numérico passo a passo** (forward, perda e backward em um mini-lote de 3 amostras reais).
- Os gradientes manuais foram comparados com o **autograd do PyTorch**, com os mesmos pesos e o mesmo mini-lote. A diferença máxima foi de **~10⁻¹⁴** (arredondamento de ponto flutuante) nas 3 arquiteturas testadas.

**Comparações controladas** (RMSE em MPa, média de 3 sementes; validação → teste):

| Experimento | Configuração | Validação | Teste |
|---|---|---|---|
| Baseline | 1 × 16, ReLU | 6,66 | 6,17 |
| C1: profundidade | 2 × 16 | 7,32 | 7,41 |
| C2: ativação | tanh | 6,08 | 5,86 |
| C3: regularização | L2 (λ = 0,01) | 6,67 | 6,08 |
| C4: taxa de aprendizado | η = 0,001 | 6,26 | 6,62 |
| C4: taxa de aprendizado | η = 0,1 | divergiu em 3/3 sementes | n/d |
| Bônus: largura | 8 / 32 neurônios | 6,97 / 6,13 | 6,98 / 6,10 |
| Bônus: inicialização (2 camadas) | ingênua / Xavier / He | 5,50 / 6,52 / 7,32 | 5,49 / n/d / 7,41 |

**Principais observações**
- **Profundidade:** a rede de 2 camadas piorou, inclusive no erro de treino. A norma dos deltas por camada não mostrou vanishing gradient severo, então o problema parece ser de otimização com gradiente descendente puro.
- **L2:** sem efeito mensurável com λ = 0,01. A penalidade equivale a ~0,3% da perda de treino, ou seja, o λ é pequeno demais nessa escala.
- **Taxa de aprendizado:** η = 0,1 diverge (a perda explode em poucos mini-lotes até virar NaN). Com η = 0,001, a validação melhora, mas o ganho **não se confirmou no teste**.
- **Inicialização:** contra a expectativa, a inicialização "ingênua" (desvio 0,01) superou a He nessa rede de 2 camadas. A explicação do notebook é hipótese, não foi testada.
- **Under/overfitting:** o baseline sofre mais de limitação de otimização (underfitting) do que de overfitting.

**Limitações** (discutidas no notebook): 3 sementes e um único split, então diferenças de poucas décimas são tendência e não prova. O L2 foi testado com um único λ. Padronização, mini-lote e critério de parada ficaram fixos.

---

## Como executar

```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
pip install numpy pandas matplotlib scikit-learn jupyter
pip install torch   # só para o Projeto 3 (conferência do gradiente)
jupyter notebook
```

- Os notebooks leem `concrete_data.csv`. Mantenha o arquivo em `data/` e ajuste o caminho no carregamento, se necessário.
- Tabelas (`.csv`) e figuras (`.png`) geradas pelos experimentos são salvas na pasta `outputs/`.
- Todas as sementes são fixas (0, 1, 2), então os resultados são reprodutíveis.

## Estrutura

```
.
├── MLP_experimentos.ipynb
├── MLP_Backpropagation.ipynb
└── README.md
```

## Tecnologias

Python · NumPy · pandas · scikit-learn · PyTorch · Matplotlib
