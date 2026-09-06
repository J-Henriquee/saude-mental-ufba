import pandas as pd
import numpy as np

# ==========================================
# CONFIGURAÇÕES E CONSTANTES
# ==========================================
# Definição centralizada dos caminhos de entrada (dados brutos) e saída (dicionário gerado).
# Centralizar isso em variáveis globais facilita a manutenção caso a estrutura de pastas mude.
RAW_PATH = "data/raw/formulario_ic_ufba_raw.csv"
OUTPUT_PATH = "docs/dicionario_forms.csv"

# Mapeamento estático de metadados (Base de Conhecimento).
# Funciona como uma tabela de "De -> Para". Isso elimina a necessidade de preencher 
# o CSV manualmente depois, padronizando nomenclaturas, agrupando perguntas por seção
# e mapeando o risco de privacidade (sensibilidade) de cada feature para os modelos.
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

# ==========================================
# FUNÇÕES DE PROCESSAMENTO
# ==========================================
def get_metadata(col_name):
    """
    Busca as propriedades de uma coluna fazendo correspondência parcial de texto (substring).
    Converte as strings para minúsculas para garantir que a busca não falhe por letras maiúsculas.
    """
    col_lower = col_name.lower()
    for key, meta in METADATA_MAP.items():
        if key.lower() in col_lower:
            return meta
    # Fallback: retorna valores padrão caso a pergunta do form não esteja no METADATA_MAP
    return {"legivel": "Nao mapeado", "secao": "Geral", "sensivel": "indefinido", "modelo": "avaliar"}

# Carregamento do dataset bruto usando Pandas
df = pd.read_csv(RAW_PATH)

# ==========================================
# FILTRO DE ESCOPO (INSTITUTO DE COMPUTAÇÃO)
# ==========================================
coluna_curso = "Selecione seu curso abaixo:" 
# Define os cursos alvo para análise
cursos_alvo = [
    "Ciência da Computação", 
    "Sistemas de Informação", 
    "Computação (Licenciatura)"
]

# Varre todas as colunas dinamicamente para encontrar qual delas contém os nomes dos cursos.
# Isso torna o código resiliente caso o título da coluna no Google Forms mude no futuro.
for col in df.columns:
    if df[col].isin(cursos_alvo).any():
        coluna_curso = col
        break

# Cria um novo DataFrame filtrando apenas os respondentes dos cursos alvo.
# O método .copy() previne o erro 'SettingWithCopyWarning' do Pandas ao manipular o novo subconjunto.
df_ic = df[df[coluna_curso].isin(cursos_alvo)].copy()

def infer_var_type(series):
    """
    Motor de inferência de tipos. Avalia os dados de uma coluna para chutar seu tipo semântico.
    Utiliza heurísticas baseadas na cardinalidade (quantidade de respostas únicas) e no dtype do Pandas.
    """
    n_unique = series.nunique()
    dtype = series.dtype
    
    if dtype == "bool": 
        return "binaria"
    if pd.api.types.is_numeric_dtype(dtype): 
        # Se for número, decide entre ordinal (poucas notas, ex: 1 a 5) ou contínua (idades, pesos, etc)
        return "ordinal" if n_unique <= 5 else "continua"
    
    # Se for texto, analisa pela quantidade de respostas diferentes
    if n_unique <= 2: return "binaria"
    if n_unique <= 10: return "categorica_nominal"
    return "texto_livre" # Muitas respostas únicas em texto geralmente indicam perguntas discursivas

def build_dictionary(dataframe):
    """
    Função principal que itera sobre o DataFrame filtrado e constrói o dicionário.
    Extrai a qualidade dos dados (nulos) e une com a base de conhecimento do METADATA_MAP.
    """
    rows = []
    for col in dataframe.columns:
        series = dataframe[col]
        n_total = len(series)
        n_missing = series.isna().sum() # Conta quantos dados faltantes (NaN) existem
        
        meta = get_metadata(col)
        
        rows.append({
            "nome_variavel": col,                            # Pergunta exata do formulário
            "nome_legivel": meta["legivel"],                 # Nome curto para usar no código/banco de dados
            "secao_formulario": meta["secao"],               # Agrupamento lógico
            "dtype_pandas": str(series.dtype),               # Tipo físico na memória
            "tipo_semantico": infer_var_type(series),        # Classificação estatística
            "n_total": n_total,
            # Calcula a porcentagem de dados faltantes arredondando para 1 casa decimal
            "pct_missing": round((n_missing / n_total) * 100, 1) if n_total > 0 else 0,
            "sensibilidade": meta["sensivel"],               # Grau de risco da informação
            "usar_no_modelo": meta["modelo"],                # Flag para feature selection
        })
    return pd.DataFrame(rows)

# ==========================================
# EXECUÇÃO PRINCIPAL DO SCRIPT
# ==========================================
# Esse bloco garante que o código só rode se o arquivo for executado diretamente,
# e não se for importado por outro módulo Python.
if __name__ == "__main__":
    # 1. Constrói a estrutura do dicionário
    dic = build_dictionary(df_ic)
    
    # 2. Exporta para CSV com 'utf-8-sig' para não quebrar acentuação no Excel
    dic.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    # 3. Logs de validação no console para checagem rápida
    print(f"Sucesso! Dicionário gerado em: {OUTPUT_PATH}")
    print(f"Total de linhas filtradas (Apenas IC): {len(df_ic)}")
    print(f"Total de variáveis mapeadas: {len(dic)}")
