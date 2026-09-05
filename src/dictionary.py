import pandas as pd
import numpy as np

RAW_PATH = "data/raw/formulario_ic_ufba_raw.csv"
OUTPUT_PATH = "docs/dicionario_forms.csv"

# Mapeamento automático de metadados para evitar preenchimento manual
METADATA_MAP = {
    "data/hora": {"legivel": "Timestamp", "secao": "Metadados", "sensivel": "baixa", "modelo": "nao"},
    "área do seu curso": {"legivel": "Area do Curso", "secao": "Academico", "sensivel": "baixa", "modelo": "nao"},
    "Selecione seu curso": {"legivel": "Curso", "secao": "Academico", "sensivel": "baixa", "modelo": "sim"},
    "semestre de ingresso": {"legivel": "Semestre de Ingresso", "secao": "Academico", "sensivel": "baixa", "modelo": "sim"},
    "seu CR": {"legivel": "CR", "secao": "Academico", "sensivel": "baixa", "modelo": "sim"},
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
    "pessoa neurodivergente": {"legivel": "Neurodivergente", "secao": "Saude Mental", "sensivel": "alta", "modelo": "sim"},
    "condições se aplicam": {"legivel": "Condicoes Neurodivergentes", "secao": "Saude Mental", "sensivel": "alta", "modelo": "sim"},
    "suporte mais eficaz": {"legivel": "Sugestao Suporte", "secao": "Feedback", "sensivel": "baixa", "modelo": "nao"}
}

def get_metadata(col_name):
    col_lower = col_name.lower()
    for key, meta in METADATA_MAP.items():
        if key.lower() in col_lower:
            return meta
    return {"legivel": "Nao mapeado", "secao": "Geral", "sensivel": "indefinido", "modelo": "avaliar"}

df = pd.read_csv(RAW_PATH)

coluna_curso = "Selecione seu curso abaixo:" 
cursos_alvo = [
    "Ciência da Computação", 
    "Sistemas de Informação", 
    "Computação (Licenciatura)"
]

for col in df.columns:
    if df[col].isin(cursos_alvo).any():
        coluna_curso = col
        break

df_ic = df[df[coluna_curso].isin(cursos_alvo)].copy()

def infer_var_type(series):
    n_unique = series.nunique()
    dtype = series.dtype
    if dtype == "bool": return "binaria"
    if pd.api.types.is_numeric_dtype(dtype): return "ordinal" if n_unique <= 5 else "continua"
    if n_unique <= 2: return "binaria"
    if n_unique <= 10: return "categorica_nominal"
    return "texto_livre"

def build_dictionary(dataframe):
    rows = []
    for col in dataframe.columns:
        series = dataframe[col]
        n_total = len(series)
        n_missing = series.isna().sum()
        
        meta = get_metadata(col)
        
        rows.append({
            "nome_variavel": col,
            "nome_legivel": meta["legivel"],
            "secao_formulario": meta["secao"],
            "dtype_pandas": str(series.dtype),
            "tipo_semantico": infer_var_type(series),
            "n_total": n_total,
            "pct_missing": round((n_missing / n_total) * 100, 1) if n_total > 0 else 0,
            "sensibilidade": meta["sensivel"],
            "usar_no_modelo": meta["modelo"],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    dic = build_dictionary(df_ic)
    dic.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Sucesso! Dicionário gerado em: {OUTPUT_PATH}")
    print(f"Total de linhas filtradas (Apenas IC): {len(df_ic)}")
    print(f"Total de variáveis mapeadas: {len(dic)}")