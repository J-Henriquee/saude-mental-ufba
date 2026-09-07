import os
import pandas as pd
import numpy as np

# ==============================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# ==============================================================================
# Ajuste os caminhos conforme a estrutura de pastas do seu repositório
RAW_DATA_PATH = "formulario_LIAO_UFBA.csv"
PROCESSED_DATA_PATH = "formulario_ic_limpo.csv"

# Cria o diretório de destino apenas se houver pastas no caminho
output_dir = os.path.dirname(PROCESSED_DATA_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# 2. CARGA E RECORTE DO ESCOPO (INSTITUTO DE COMPUTAÇÃO)
# ==============================================================================
print("[INFO] Iniciando pipeline de limpeza e pré-processamento...")
df_raw = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")

# Cursos do escopo da análise
cursos_alvo = [
    "Ciência da Computação",
    "Sistemas de Informação",
    "Computação (Licenciatura)"
]

# Identifica dinamicamente qual coluna contém as opções de curso do IC
col_curso_raw = None
for col in df_raw.columns:
    if df_raw[col].isin(cursos_alvo).any():
        col_curso_raw = col
        break

if not col_curso_raw:
    raise ValueError("Coluna de cursos do IC não foi identificada na planilha bruta.")

# Filtra estritamente os respondentes do IC
df_clean = df_raw[df_raw[col_curso_raw].isin(cursos_alvo)].copy()
print(f"[INFO] Filtro de escopo aplicado: {len(df_clean)} respondentes selecionados.")

# Remove colunas fantasmas (seções vazias de outras áreas não preenchidas pelo IC)
cols_fantasmas = [
    c for c in df_clean.columns 
    if c.startswith("Unnamed:") and c != col_curso_raw and df_clean[c].isna().all()
]
df_clean.drop(columns=cols_fantasmas, inplace=True)
print(f"[INFO] {len(cols_fantasmas)} colunas vazias de outros departamentos descartadas.")

# ==============================================================================
# 3. RENOMEAÇÃO SEMÂNTICA DAS VARIÁVEIS (PADRÃO SNAKE_CASE)
# ==============================================================================
def find_col(substring):
    """Localiza o nome da coluna no DataFrame através de busca parcial insensível a maiúsculas."""
    matches = [c for c in df_clean.columns if substring.lower() in c.lower()]
    return matches[0] if matches else None

rename_mapping = {
    find_col("data/hora"): "timestamp",
    find_col("área do seu curso"): "area_curso",
    col_curso_raw: "curso",
    find_col("semestre de ingresso"): "semestre_ingresso_bruto",
    find_col("seu CR"): "cr",
    find_col("ensino pré-faculdade"): "tipo_escola_base",
    find_col("forma de ingresso"): "forma_ingresso",
    find_col("identidade de gênero"): "genero",
    find_col("autodeclara"): "raca_cor",
    find_col("quais turnos"): "turnos_estudo",
    find_col("carga acadêmica"): "percepcao_carga_academica",
    find_col("sobrecarregado"): "frequencia_sobrecarga",
    find_col("tempo efetivo de estudo"): "horas_estudo_diario",
    find_col("tempo médio"): "tempo_deslocamento_min",
    find_col("tempo do seu dia"): "tempo_na_ufba",
    find_col("trabalha?"): "trabalha",
    find_col("extracurriculares"): "atividades_extracurriculares",
    find_col("acompanhar as disciplinas"): "dificuldade_academica",
    find_col("desempenho reflete no seu bem estar"): "impacto_bem_estar",
    find_col("dificuldades específicas"): "dificuldades_especificas",
    find_col("abandonar seu curso"): "risco_evasao",
    find_col("maior dificuldade que você enfrenta"): "maior_dificuldade_aberta",
    find_col("encontrou esse formulário"): "origem_resposta",
    find_col("acompanhamento profissional"): "acompanhamento_psicologico",
    find_col("modalidade das suas sessões"): "modalidade_terapia",
    find_col("frequência você realiza"): "frequencia_terapia",
    find_col("pessoa neurodivergente"): "neurodivergente",
    find_col("condições se aplicam"): "condicoes_neurodivergentes",
    find_col("suporte mais eficaz"): "sugestao_suporte_aberta",
}

# Aplica renomeação descartando eventuais valores nulos de busca
df_clean.rename(columns={k: v for k, v in rename_mapping.items() if k is not None}, inplace=True)

# ==============================================================================
# 4. TRATAMENTO DE TIPAGEM E ENGENHARIA DE ATRIBUTOS
# ==============================================================================
# Conversão do Timestamp para datetime padrão
df_clean["timestamp"] = pd.to_datetime(
    df_clean["timestamp"].astype(str).str.replace(" GMT-3", "", regex=False),
    format="%Y/%m/%d %I:%M:%S %p",
    errors="coerce"
)

# Decomposição do semestre de ingresso (Ex.: 2024.1 -> Ano: 2024, Semestre: 1)
df_clean["semestre_ingresso_str"] = df_clean["semestre_ingresso_bruto"].astype(str)
df_clean["ano_ingresso"] = df_clean["semestre_ingresso_str"].apply(
    lambda x: int(x.split(".")[0]) if "." in x else np.nan
).astype("Int64")

df_clean["semestre_letivo_ingresso"] = df_clean["semestre_ingresso_str"].apply(
    lambda x: int(x.split(".")[1]) if "." in x else np.nan
).astype("Int64")

# Indicador de fornecimento voluntário de CR
df_clean["cr_informado"] = df_clean["cr"].notna().astype(int)

# Binarização de respostas dicotômicas ("Sim" / "Não")
colunas_sim_nao = [
    "atividades_extracurriculares",
    "dificuldade_academica",
    "impacto_bem_estar",
    "risco_evasao"
]
for col in colunas_sim_nao:
    df_clean[col + "_bin"] = df_clean[col].map({"Sim": 1, "Não": 0}).astype("Int64")

# ==============================================================================
# 5. DESMEMBRAMENTO MULTI-LABEL (CAIXAS DE SELEÇÃO COM ';')
# ==============================================================================
# 5.1 Dificuldades Acadêmicas Específicas
dificuldades_dummies = df_clean["dificuldades_especificas"].str.get_dummies(sep=";")
dificuldades_dummies.columns = [
    "dif_" + c.strip().lower()
    .replace(" ", "_")
    .replace("/", "_")
    .replace("ã", "a")
    .replace("ç", "c")
    .replace("õ", "o")
    .replace("é", "e")
    for c in dificuldades_dummies.columns
]

# 5.2 Condições de Neurodivergência
condicoes_dummies = df_clean["condicoes_neurodivergentes"].str.get_dummies(sep=";")
condicoes_dummies.columns = [
    "cond_" + c.strip().lower()
    .replace(" ", "_")
    .replace("/", "_")
    .replace("(", "")
    .replace(")", "")
    .replace("-", "_")
    .replace("ã", "a")
    .replace("é", "e")
    .replace("ó", "o")
    for c in condicoes_dummies.columns
]

# Consolidação do Dataset Final
df_final = pd.concat([df_clean, dificuldades_dummies, condicoes_dummies], axis=1)

# ==============================================================================
# 6. EXPORTAÇÃO E LOG DE CONCLUSÃO
# ==============================================================================
df_final.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")

print("=" * 70)
print(f"[SUCESSO] Dataset limpo e pré-processado exportado com sucesso!")
print(f"Destino: {PROCESSED_DATA_PATH}")
print(f"Dimensões finais: {df_final.shape[0]} linhas x {df_final.shape[1]} colunas.")
print(f"Variáveis dummies de Dificuldade geradas: {len(dificuldades_dummies.columns)}")
print(f"Variáveis dummies de Neurodivergência geradas: {len(condicoes_dummies.columns)}")
print("=" * 70)