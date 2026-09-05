# saude-mental-ufba

Projeto de pesquisa da **Liga Acadêmica de IA e Otimização (LIAO) — UFBA**  
Análise descritiva e exploratória de dados sobre saúde mental e risco de evasão em estudantes neurodivergentes.

> **Escopo:** Análise estatística descritiva com visualização em Looker Studio. Sem modelagem preditiva.  
> **Produto final:** Artigo científico + dashboard público de resultados.

---

## Estrutura do repositório

```
saude-mental-ufba/
│
├── data/
│   ├── raw/                  # Dataset original exportado do Google Forms (não versionar dados sensíveis)
│   └── processed/            # Dataset limpo, gerado pela TASK-03
│
├── docs/
│   ├── dicionario_dados.csv  # Dicionário de variáveis — output da TASK-02
│   ├── log_limpeza.md        # Decisões de pré-processamento — output da TASK-03
│   └── referencias/          # Referências bibliográficas fichadas — TASK-05
│
├── notebooks/
│   ├── 02_dicionario.ipynb   # Geração do dicionário de dados
│   ├── 03_limpeza.ipynb      # Pré-processamento e limpeza
│   └── 04_eda.ipynb          # Análise descritiva e exploratória
│
├── src/
│   ├── dictionary.py         # Script de geração automática do dicionário
│   └── cleaning.py           # Script de limpeza do dataset
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

- Python 3.10+
- pip

---

## Setup local

### 1. Clonar o repositório

```bash
git clone https://github.com/J-Henriquee/saude-mental-ufba.git
cd saude-mental-ufba
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Colocar o dataset bruto

Exporte as respostas do Google Forms como CSV e salve em:

```
data/raw/formulario_ic_ufba_raw.csv
```

---

## Como executar

### TASK-02 — Gerar o dicionário de dados

```bash
python src/dictionary.py
```

Gera `docs/dicionario_dados.csv`. Após rodar, preencha manualmente as colunas:
- `nome_legivel` — nome humano da variável
- `secao_formulario` — a qual seção do Forms pertence
- `sensibilidade` — `alta` (saúde mental, diagnósticos) / `media` / `baixa`
- `usar_no_estudo` — `sim` / `nao` / `avaliar`

### TASK-03 — Limpar o dataset

```bash
python src/cleaning.py
```

Gera `data/processed/dataset_limpo.csv`. Todas as decisões de limpeza são registradas em `docs/log_limpeza.md`.

### TASK-04 — Análise exploratória

Abra o notebook no Jupyter:

```bash
jupyter notebook notebooks/04_eda.ipynb
```

---

## Dependências Python

```
pandas
numpy
scipy
matplotlib
seaborn
jupyter
openpyxl
```

> O `requirements.txt` já está no repositório. Se precisar atualizar após instalar algo novo:
> ```bash
> pip freeze > requirements.txt
> ```

---

## Fluxo das tasks

```
TASK-01 ✅ Mapeamento de fontes externas (Acabado)
   ↓
TASK-02   Auditoria e dicionário de dados
   ↓
TASK-03   Limpeza e pré-processamento do dataset
   ↓
TASK-04   Análise estatística descritiva (EDA) + Looker Studio
   ↓
TASK-05   Redação do manuscrito e submissão
```

---

## Convenções do projeto

**Commits:** seguir o padrão `feat:`, `fix:`, `docs:`, `chore:`  
Exemplos:
```
feat: adiciona script de geração do dicionário
docs: atualiza log de limpeza com decisão de missing values
chore: atualiza requirements.txt
```

**Branches:** uma branch por task  
```
task/02-dicionario-dados
task/03-limpeza-dataset
task/04-eda
```

**Notebooks:** numerar pelo ID da task (`02_`, `03_`, `04_`) e manter células com saída limpa antes de commitar.

---

## Equipe e contribuições (CRediT)

| Membro | Tasks | Papel CRediT |
|---|---|---|
| Danielle Santos de Souza | TASK-01, TASK-02 | Investigação, Curadoria de Dados |
| José Henrique do Espírito Santo | TASK-02, TASK-03 | Curadoria de Dados, Software |
| — | TASK-04 | Análise Formal |
| — | TASK-05 | Redação — Original |

> A tabela de autoria final com CPs será preenchida ao término do projeto.

---

## Dados e privacidade
Os dados desse projeto foram anonimizados respeitando a leis LGPD no próprio forms

---

## Licença

Uso acadêmico restrito — LIAO / UFBA. Não redistribuir sem autorização dos autores.