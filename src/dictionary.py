import os
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURAÇÕES E CAMINHOS PADRONIZADOS
# ==============================================================================
RAW_PATH = "data/raw/formulario_ic_ufba_raw.csv"
OUTPUT_PATH = "docs/dicionario_forms.csv"

# Garante que a pasta de destino exista
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Mapeamento estático de metadados
METADATA_MAP = {
    "data/hora": {"legivel": "Timestamp", "secao": "Metadados", "sensivel": "baixa", "modelo": "nao"},
    "área do seu curso": {"legivel": "Area do Curso", "secao": "Academico", "sensivel": "baixa", "modelo": "nao"},
    "semestre de ingresso": {"legivel": "Semestre de Ingresso", "secao": "Academico", "sensivel": "baixa", "modelo": "sim"},
    "seu cr": {"legivel": "CR", "secao": "Academico", "sensivel": "baixa", "modelo": "sim"},
    "ensino pré-faculdade": {"legivel": "Tipo de Escola Base", "secao": "Demografico", "sensivel": "baixa", "modelo": "sim"},
    "forma de ingresso": {"legivel": "Forma de Ingresso", "secao": "Demografico", "sensivel": "baixa", "modelo": "sim"},
    "identidade de gênero": {"legivel": "Genero", "secao": "Demografico", "sensivel": "baixa", "modelo": "sim"},
    "autodeclara": {"legivel": "Raca/Cor", "secao": "Demografico", "sensivel": "baixa", "modelo": "sim"},
    "quais turnos": {"legivel": "Turnos de Estudo", "secao": "Rotina", "sensivel": "baixa", "modelo": "sim"},
    "carga acadêmica": {"legivel": "Percepcao Carga Academica", "secao": "Rotina", "sensivel": "media", "modelo": "sim"},
    "sobrecarregado": {"legivel": "Frequencia Sobrecarga", "secao": "Saude Mental", "sensivel": "media", "modelo": "sim"},
    "tempo efetivo de estudo": {"legivel": "Horas Estudo Diario", "secao": "Rotina", "sensivel": "baixa", "modelo": "sim"},
    "tempo médio (em minutos) que leva": {"legivel": "Tempo Deslocamento", "secao": "Rotina", "sensivel": "baixa", "modelo": "sim"},
    "tempo do seu dia você passa": {"legivel": "Tempo na UFBA", "secao": "Rotina", "sensivel": "baixa", "modelo": "sim"},
    "trabalha?": {"legivel": "Trabalha", "secao": "Socioeconomico", "sensivel": "baixa", "modelo": "sim"},
    "extracurriculares": {"legivel": "Atividades Extracurriculares", "secao": "Academico", "sensivel": "baixa", "modelo": "sim"},
    "dificuldade em acompanhar": {"legivel": "Dificuldade Academica", "secao": "Academico", "sensivel": "media", "modelo": "sim"},
    "desempenho reflete no seu bem estar": {"legivel": "Impacto Bem Estar", "secao": "Saude Mental", "sensivel": "media", "modelo": "sim"},
    "dificuldades específicas": {"legivel": "Dificuldades Especificas", "secao": "Academico", "sensivel": "baixa", "modelo": "avaliar"},
    "abandonar seu curso": {"legivel": "Risco Evasao", "secao": "Academico", "sensivel": "alta", "modelo": "sim"},
    "maior dificuldade": {"legivel": "Maior Dificuldade Aberta", "secao": "Academico", "sensivel": "baixa", "modelo": "nao"},
    "encontrou esse formulário": {"legivel": "Origem Resposta", "secao": "Metadados", "sensivel": "baixa", "modelo": "nao"},
    "acompanhamento profissional": {"legivel": "Acompanhamento Psicologico", "secao": "Saude Mental", "sensivel": "alta", "modelo": "sim"},
    "modalidade das suas sessões": {"legivel": "Modalidade Terapia", "secao": "Saude Mental", "sensivel": "alta", "modelo": "avaliar"},
    "frequência você realiza": {"legivel": "Frequencia Terapia", "secao": "Saude Mental", "sensivel": "alta", "modelo": "avaliar"},
    "você é uma pessoa neurodivergente": {"legivel": "Neurodivergente", "secao": "Saude Mental", "sensivel": "alta", "modelo": "sim"},
    "condições se aplicam": {"legivel": "Condicoes Neurodivergentes", "secao": "Saude Mental", "sensivel": "alta", "modelo": "sim"},
    "suporte mais eficaz": {"legivel": "Sugestao Suporte", "secao": "Feedback", "sensivel": "baixa", "modelo": "nao"}
}

def get_metadata(col_name, coluna_curso_detectada=None):
    if coluna_curso_detectada and col_name == coluna_curso_detectada:
        return {"legivel": "Curso", "secao": "Academico", "sensivel": "baixa", "modelo": "sim"}

    col_lower = col_name.lower()
    for key in sorted(METADATA_MAP.keys(), key=len, reverse=True):
        if key.lower() in col_lower:
            return METADATA_MAP[key]

    return {"legivel": "Nao mapeado", "secao": "Geral", "sensivel": "indefinido", "modelo": "avaliar"}

def infer_var_type(series, col_name=""):
    n_unique = series.nunique()
    col_lower = col_name.lower()

    if "data/hora" in col_lower or "timestamp" in col_lower or pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if n_unique <= 1:
        return "constante"
    if "semestre" in col_lower or "ano" in col_lower:
        return "temporal_ordinal"
    if series.dtype == "bool":
        return "binaria"
    if any(k in col_lower for k in ["dificuldades específicas", "condições se aplicam", "marque todas"]):
        return "categorica_multilabel"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "ordinal" if n_unique <= 5 else "continua"
    if n_unique == 2:
        return "binaria"
    if n_unique <= 10:
        return "categorica_nominal"
    return "texto_livre"

def build_dictionary(dataframe, coluna_curso=None):
    rows = []
    for col in dataframe.columns:
        series = dataframe[col]
        n_total = len(series)
        n_missing = series.isna().sum()
        meta = get_metadata(col, coluna_curso_detectada=coluna_curso)

        rows.append({
            "nome_variavel": col,
            "nome_legivel": meta["legivel"],
            "secao_formulario": meta["secao"],
            "dtype_pandas": str(series.dtype),
            "tipo_semantico": infer_var_type(series, col),
            "n_total": n_total,
            "pct_missing": round((n_missing / n_total) * 100, 1) if n_total > 0 else 0,
            "sensibilidade": meta["sensivel"],
            "usar_no_modelo": meta["modelo"],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    cursos_alvo = ["Ciência da Computação", "Sistemas de Informação", "Computação (Licenciatura)"]

    coluna_curso = None
    for col in df.columns:
        if df[col].isin(cursos_alvo).any():
            coluna_curso = col
            break

    if not coluna_curso:
        raise ValueError("Coluna de cursos alvo não encontrada no dataset bruto.")

    df_ic = df[df[coluna_curso].isin(cursos_alvo)].copy()
    cols_fantasmas = [c for c in df_ic.columns if c.startswith("Unnamed:") and c != coluna_curso and df_ic[c].isna().all()]
    df_ic.drop(columns=cols_fantasmas, inplace=True)

    dic = build_dictionary(df_ic, coluna_curso=coluna_curso)
    dic.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Sucesso! Dicionário gerado em: {OUTPUT_PATH}")
    print(f"Total de respondentes (Apenas IC): {len(df_ic)}")
    print(f"Colunas ativas catalogadas: {len(dic)}")