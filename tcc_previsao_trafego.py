# -*- coding: utf-8 -*-
"""
================================================================================
 MODELOS ESTATÍSTICOS E DE APRENDIZADO DE MÁQUINA NA PREVISÃO DE TRÁFEGO
 EM RODOVIAS CONCEDIDAS — rotina completa do TCC (MBA em Data Science e
 Analytics — USP/Esalq)
 Repositório: https://github.com/Vasconcelosrafael23/TCC_USP
--------------------------------------------------------------------------------
 ETAPAS
   1. Leitura dos CSVs anuais da ANTT ("Volume de Tráfego nas Praças de
      Pedágio"), com os formatos de data dd/mm/aaaa (até 2023) e mm/aaaa
      (arquivo mensal consolidado de 2024), e corte da série em nov/2024.
   2. Série mensal da praça de Vargem (SP), com auditoria de completude.
   3. Decomposição STL robusta, forças de tendência e sazonalidade, ADF e KPSS.
   4. Holt-Winters, SARIMA, Prophet e XGBoost contra o naïve sazonal, com
      validação por origem móvel em h = 1 e h = 12.
   5. MAE, RMSE, MAPE e MASE; Diebold-Mariano par a par sob perda quadrática e
      absoluta; Ljung-Box e teste de viés sobre os erros do melhor modelo.
   6. Sensibilidade à quebra pandêmica (mar-dez/2020 ajustados pela STL
      estimada só no treino).
   7. Previsão de 12 meses a partir de nov/2024 e planilha serie_prevista.xlsx.
   8. Análises complementares: correção de Holm; testes de raiz unitária
      sazonal OCSB e Canova-Hansen; SARIMA com D = 0; viés de 1 p.p. na taxa
      de crescimento do tráfego (o exercício de PIB x elasticidade com dados de
      2025 permanece no código, desativado).
   0. Antes de tudo, a configuração do Prophet é escolhida só com o treino
      (validação interna nos últimos 24 meses do treino; selecao_prophet.txt).
   9. Figuras do TCC (Figura 1: STL; Figura 2: janela de teste; Figura 3: erro
      relativo ao naïve por passo; Figura 4: previsão com intervalo de 95%).
  10. Conferência dos números citados no texto (numeros_do_texto.txt).

 DADOS
   Coloque os CSVs anuais da ANTT em ./dados/ (2010 a 2024 e o consolidado de
   2025, usado só no item 8). Sem eles, a rotina tenta baixar de
   dados.antt.gov.br.

 COMO RODAR
   python tcc_previsao_trafego.py
   Saídas: resultados_para_texto.txt, analises_complementares.txt, numeros_do_texto.txt,
   serie_prevista.xlsx, metricas_*.csv e figuras PNG na pasta do projeto.

 AMBIENTE EM QUE OS RESULTADOS FORAM REPRODUZIDOS
   Python 3.12.3, statsmodels 0.15.0, scikit-learn 1.8.0, xgboost 3.4.1,
   prophet 1.4.0, scipy 1.17.1, pandas 3.0.2, numpy 2.4.4, pmdarima 2.1.1.
   Tempo de execução: cerca de 10 minutos.
================================================================================
"""

# --- Instalação automática das bibliotecas (roda uma vez, na primeira execução)
#     Equivalente, no terminal, a:
#     pip install pandas numpy matplotlib statsmodels scikit-learn xgboost
#                 prophet scipy openpyxl holidays pmdarima
import subprocess, sys
_PACOTES = ["pandas", "numpy", "matplotlib", "statsmodels", "scikit-learn",
            "xgboost", "prophet", "scipy", "openpyxl", "holidays", "pmdarima"]
try:
    import pandas, numpy, matplotlib, statsmodels, sklearn, xgboost  # noqa
    import prophet, scipy, openpyxl, holidays, pmdarima              # noqa
except ImportError:
    print("Instalando bibliotecas necessárias (primeira execução)...")
    # em ambientes 'externally-managed', acrescente '--break-system-packages'
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_PACOTES])

import os, glob, warnings, io, textwrap
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet
import holidays as pyholidays


# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
PASTA_DADOS   = "dados"
PASTA_UPLOADS = "/mnt/user-data/uploads"

# Caminho opcional de um CSV com as séries mensais já extraídas (colunas
# 'data', 'total', 'leves', 'pesados'), tal como produzido por
# exportar_series.py. Se preenchido e existente, a rotina o utiliza e dispensa
# a leitura dos CSVs brutos da ANTT. Deixe vazio para o fluxo normal.
ARQUIVO_SERIES = ""

SAZONALIDADE  = 12          # ciclo anual em série mensal
N_TESTE       = 24          # meses reservados para teste (janela de avaliação)
HORIZONTE_FUT = 12          # meses da previsão futura reportada
SEED          = 42
DATA_FIM      = "2024-11-01"   # [ajuste] fim da série (integridade)
np.random.seed(SEED)

# Praça-alvo. O filtro é por 'contém', sem acento, para tolerar grafias.
CONCESSIONARIA_ALVO = "FERNAO DIAS"     # BR-381, MG/SP
PRACA_ALVO          = "Vargem"          # vazio => escolhe a de série mais completa

# Se True, o nome da concessionária não aparece em figuras, console nem Excel.
# Mantenha coerente com o que o texto do TCC afirma sobre anonimização.
ANONIMIZAR = True

# Janela da quebra estrutural pandêmica (usada só no cenário de sensibilidade).
COVID_INICIO = "2020-03-01"
COVID_FIM    = "2020-12-01"

# Blocos opcionais — desligue para uma passada rápida.
RODAR_H12                 = True    # backtest de horizonte longo (h = 12)
RODAR_SENSIBILIDADE_COVID = True    # repete tudo na série ajustada
RODAR_SEGMENTOS           = False   # leves e pesados (não reportado no TCC)
RODAR_COMPARACAO_FERIADOS = False   # comparação na janela de teste desativada; a escolha é feita no treino

# Tratamento de feriados no Prophet: "mensal" | "nativo" | "nenhum"
PROPHET_FERIADOS = "nenhum"   # redefinido por selecionar_config_prophet() a partir do treino

# Nomes de exibição (usados em tabelas e figuras — sem underscore).
NAIVE   = "Naïve sazonal"
HW      = "Holt-Winters"
SARIMA  = "SARIMA"
PROPHET = "Prophet"
XGB     = "XGBoost"
ORDEM_EXIBICAO = [HW, PROPHET, SARIMA, XGB, NAIVE]

URLS = {
 2010:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/4c1c6cde-61a3-4704-9b01-4bb6e75b874d/download/volume-trafego-praca-pedagio-2010.csv",
 2011:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/2c1f7d1f-2fd4-451d-bddf-590e0d5d41ba/download/volume-trafego-praca-pedagio-2011.csv",
 2012:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/25f15485-0a98-407d-a890-e985a60bfe21/download/volume-trafego-praca-pedagio-2012.csv",
 2013:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/a6d1bf5c-d836-4c2b-b72d-12dfba093457/download/volume-trafego-praca-pedagio-2013.csv",
 2014:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/f0a75a7d-6d3e-4fcb-a7f5-fdac1c4e4952/download/volume-trafego-praca-pedagio-2014.csv",
 2015:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/fff18b6a-08fa-4fb2-833c-d69f7e74247c/download/volume-trafego-praca-pedagio-2015.csv",
 2016:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/f15a01e4-8ad5-454c-83e3-508df5c617aa/download/volume-trafego-praca-pedagio-2016.csv",
 2017:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/41b5b002-86b1-42eb-a466-323def65542d/download/volume-trafego-praca-pedagio-2017.csv",
 2018:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/2441ec53-bc7c-4142-8da3-79f025a6fd0a/download/volume-trafego-praca-pedagio-2018.csv",
 2019:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/e48ac27c-f02e-435f-a4e3-ac35f287a3c9/download/volume-trafego-praca-pedagio-2019.csv",
 2020:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/ec91ba23-5cc4-44e7-92a0-2f73b8095b4f/download/volume-trafego-praca-pedagio-2020.csv",
 2021:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/07e6461d-01f3-4f17-ac9e-f556aaf05d7a/download/volume-trafego-praca-pedagio-2021.csv",
 2022:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/c01c6e55-1bbb-457f-9e84-40fd2f18c80a/download/volume-trafego-praca-pedagio-2022.csv",
 2023:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/e37feed2-816d-4541-897a-8c89214f6a13/download/volume-trafego-praca-pedagio-2023.csv",
 2024:"https://dados.antt.gov.br/dataset/5bf70ec3-b24e-4f73-99a0-78b200f5e915/resource/7f622637-a442-4f3c-8360-aebc41806ca3/download/volume-trafego-praca-pedagio-2024.csv",
}

MESES_PT = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,
            "jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}

# Acumulador do relatório final (espelha o console em resultados_para_texto.txt).
_RELATORIO = []
def diga(txt=""):
    """Imprime no console e guarda para o arquivo de transcrição."""
    print(txt)
    _RELATORIO.append(str(txt))


def br(x, dec=0):
    """Formata número no padrão brasileiro: ponto de milhar, vírgula decimal.
    Escrito assim para evitar o erro clássico de aplicar .replace() na linha
    inteira, que corrompe pontos de outros usos (leaders, siglas, rótulos)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    s = f"{x:,.{dec}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def tabela_br(df, dec=3):
    """to_string() com números no padrão brasileiro. Colunas cujos valores são
    todos inteiros (posições, contagens) saem sem casas decimais."""
    d = df.copy()
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]):
            continue
        v = d[c].dropna()
        inteira = len(v) > 0 and bool(np.all(np.mod(v.values, 1) == 0))
        casas = 0 if inteira else dec
        d[c] = d[c].map(lambda x: br(x, casas) if pd.notna(x) else "")
    return d.to_string()


def fmt_ordem(ordem):
    (p_, d_, q_), (P_, D_, Q_, s_) = ordem
    return f"SARIMA({p_},{d_},{q_})({P_},{D_},{Q_})[{s_}]"



# ==============================================================================
# 1) OBTENÇÃO DOS DADOS
# ==============================================================================
def baixar_dados():
    """Baixa os CSVs para ./dados (requer 'dados.antt.gov.br' liberado)."""
    import urllib.request
    os.makedirs(PASTA_DADOS, exist_ok=True)
    for ano, url in URLS.items():
        destino = os.path.join(PASTA_DADOS, f"volume-{ano}.csv")
        if os.path.exists(destino):
            continue
        print(f"   baixando {ano}...")
        urllib.request.urlretrieve(url, destino)
    print("   download concluído.")


def localizar_csvs():
    """Encontra os CSVs em ./dados ou em /mnt/user-data/uploads."""
    arqs = sorted(glob.glob(os.path.join(PASTA_DADOS, "*.csv")))
    if not arqs and os.path.isdir(PASTA_UPLOADS):
        arqs = sorted(glob.glob(os.path.join(PASTA_UPLOADS, "*trafego*.csv")))
    return arqs


# ==============================================================================
# 2) LEITURA E CONSTRUÇÃO DA SÉRIE
# ==============================================================================
def _sem_acento(s):
    return (s.astype(str).str.normalize("NFKD")
             .str.encode("ascii", "ignore").str.decode("ascii").str.upper())


def _parse_mes_ano(serie):
    """Converte mes_ano em datetime (1º dia do mês), tolerando os dois formatos
    presentes na base: 'DD/MM/AAAA' (anos antigos) e 'mmm-aa' (anos recentes)."""
    s = serie.astype(str).str.strip().str.lower()
    dt = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    # [ajuste] formato 'MM/AAAA' do arquivo mensal consolidado de 2024
    m2 = dt.isna()
    dt.loc[m2] = pd.to_datetime(s[m2], format="%m/%Y", errors="coerce")
    faltantes = dt.isna()
    if faltantes.any():
        def conv(txt):
            try:
                m, a = txt.replace("/", "-").split("-")[:2]
                mes = MESES_PT.get(m[:3], None)
                if mes is None:
                    return pd.NaT
                ano = int(a); ano = ano + 2000 if ano < 100 else ano
                return pd.Timestamp(year=ano, month=mes, day=1)
            except Exception:
                return pd.NaT
        dt.loc[faltantes] = s[faltantes].map(conv)
    return dt.dt.to_period("M").dt.to_timestamp()


def carregar_tudo(arquivos):
    """Lê e concatena todos os CSVs, com colunas tipadas e limpas."""
    frames, descartes = [], 0
    for a in arquivos:
        df = pd.read_csv(a, sep=";", encoding="latin-1", decimal=",",
                         thousands=".", dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        df["volume_total"] = pd.to_numeric(
            df["volume_total"].astype(str).str.replace(".", "", regex=False)
                                          .str.replace(",", ".", regex=False),
            errors="coerce")
        df["data"] = _parse_mes_ano(df["mes_ano"])
        n0 = len(df)
        df = df.dropna(subset=["data", "volume_total"])
        descartes += n0 - len(df)
        frames.append(df[["concessionaria", "praca", "tipo_de_veiculo",
                          "volume_total", "data"]])
    base = pd.concat(frames, ignore_index=True)
    # [ajuste] corte da série no último mês de integridade confirmada
    base = base[base["data"] <= pd.Timestamp(DATA_FIM)]
    diga(f"    registros lidos: {br(len(base) + descartes)}")
    diga(f"    registros descartados por data/volume ilegível: {br(descartes)}")
    return base


def carregar_series_prontas(caminho):
    """Lê o CSV de séries mensais já extraídas (exportar_series.py).
    Devolve dict classe -> Series mensal SEM interpolação (NaN preservados)."""
    d = pd.read_csv(caminho, parse_dates=["data"]).set_index("data").asfreq("MS")
    return {c: d[c].astype(float) for c in d.columns}


def serie_de_prontas(prontas, classe="total", verboso=True):
    """Mesmo contrato de construir_serie(), a partir das séries já extraídas."""
    if classe not in prontas:
        raise ValueError(f"classe '{classe}' ausente no arquivo de séries")
    bruta = prontas[classe]
    faltantes = bruta[bruta.isna()].index
    n = len(bruta)
    inicio_teste = bruta.index[n - N_TESTE] if n > N_TESTE else bruta.index[0]
    falt_no_teste = [d for d in faltantes if d >= inicio_teste]
    serie = bruta.interpolate(limit_direction="both")
    serie.name = f"volume_{classe}"
    diag = {"n_meses": n, "n_faltantes": len(faltantes),
            "meses_faltantes": [d.strftime("%Y-%m") for d in faltantes],
            "faltantes_no_teste": [d.strftime("%Y-%m") for d in falt_no_teste]}
    praca = PRACA_ALVO or "(praça do arquivo de séries)"
    if verboso:
        diga(f"    praça selecionada: {praca}   |   classe: {classe}")
        diga(f"    série: {n} meses ({bruta.index.min():%Y-%m} a "
             f"{bruta.index.max():%Y-%m})")
        if diag["n_faltantes"] == 0:
            diga("    observações ausentes: nenhuma (série completa).")
        else:
            diga(f"    observações ausentes: {diag['n_faltantes']} "
                 f"-> {', '.join(diag['meses_faltantes'])}")
            diga("      tratadas por interpolação linear.")
            if falt_no_teste:
                diga("      [!] ATENÇÃO: há mês ausente DENTRO da janela de "
                     "teste.")
            else:
                diga("      todas anteriores à janela de teste.")
    return serie, praca, diag


def construir_serie(base, classe="total", verboso=True):
    """Filtra concessionária/praça-alvo e devolve a série mensal + diagnóstico
    de completude (quantos meses faltavam, quais, e se caem no teste)."""
    b = base.copy()
    b["conc_norm"] = _sem_acento(b["concessionaria"])
    alvo = _sem_acento(pd.Series([CONCESSIONARIA_ALVO]))[0]
    b = b[b["conc_norm"].str.contains(alvo)]
    if b.empty:
        raise ValueError("Concessionária-alvo não encontrada. "
                         "Ajuste CONCESSIONARIA_ALVO.")

    tv = b["tipo_de_veiculo"].astype(str).str.lower()
    if classe == "leves":
        b = b[tv.str.contains("passeio|moto")]
    elif classe == "pesados":
        b = b[tv.str.contains("comerc")]

    if PRACA_ALVO:
        b = b[b["praca"].astype(str).str.contains(PRACA_ALVO, case=False)]
        praca = PRACA_ALVO
    else:
        praca = b.groupby("praca")["data"].nunique().idxmax()
        b = b[b["praca"] == praca]
    if b.empty:
        raise ValueError(f"Nenhum registro para praça/classe '{praca}/{classe}'.")

    bruta = b.groupby("data")["volume_total"].sum().sort_index().asfreq("MS")

    # --- diagnóstico de completude (ANTES de interpolar) ----------------------
    faltantes = bruta[bruta.isna()].index
    n = len(bruta)
    inicio_teste = bruta.index[n - N_TESTE] if n > N_TESTE else bruta.index[0]
    falt_no_teste = [d for d in faltantes if d >= inicio_teste]

    serie = bruta.interpolate(limit_direction="both")
    serie.name = f"volume_{classe}"

    diag = {"n_meses": n,
            "n_faltantes": len(faltantes),
            "meses_faltantes": [d.strftime("%Y-%m") for d in faltantes],
            "faltantes_no_teste": [d.strftime("%Y-%m") for d in falt_no_teste]}

    if verboso:
        diga(f"    praça selecionada: {praca}   |   classe: {classe}")
        diga(f"    série: {n} meses ({bruta.index.min():%Y-%m} a "
             f"{bruta.index.max():%Y-%m})")
        if diag["n_faltantes"] == 0:
            diga("    observações ausentes: nenhuma (série completa).")
        else:
            diga(f"    observações ausentes: {diag['n_faltantes']} "
                 f"-> {', '.join(diag['meses_faltantes'])}")
            diga("      tratadas por interpolação linear.")
            if falt_no_teste:
                diga("      [!] ATENÇÃO: há mês ausente DENTRO da janela de "
                     "teste " + f"({', '.join(diag['faltantes_no_teste'])}). "
                     "A interpolação usa informação posterior; declare isso "
                     "como limitação no texto.")
            else:
                diga("      todas anteriores à janela de teste — sem uso de "
                     "informação futura.")
    return serie, praca, diag


def ajustar_quebra_covid(serie, n_teste=N_TESTE):
    """Substitui os meses da janela pandêmica pelo sinal 'tendência + sazonal'
    de uma decomposição STL robusta, removendo apenas a componente irregular.

    O STL é estimado SOMENTE no trecho de treino, de modo que o ajuste não
    incorpora informação da janela de teste. O resultado é aplicado de forma
    idêntica a todos os modelos, preservando a equidade da comparação."""
    treino = serie.iloc[:len(serie) - n_teste]
    st = STL(treino, period=SAZONALIDADE, robust=True).fit()
    sinal = pd.Series(np.asarray(st.trend) + np.asarray(st.seasonal),
                      index=treino.index)
    ini, fim = pd.Timestamp(COVID_INICIO), pd.Timestamp(COVID_FIM)
    alvo = [d for d in serie.index if ini <= d <= fim and d in sinal.index]
    aj = serie.copy()
    aj.loc[alvo] = sinal.loc[alvo].values
    aj.name = serie.name + "_ajustada"
    return aj, alvo


# ==============================================================================
# 3) MÉTRICAS E TESTES
# ==============================================================================
def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return np.mean(np.abs((y - p) / y)) * 100


def escala_mase(treino, m=SAZONALIDADE):
    """Erro médio absoluto do naïve sazonal DENTRO do treino — denominador do
    MASE (Hyndman & Koehler, 2006)."""
    tr = np.asarray(treino, float)
    return np.mean(np.abs(tr[m:] - tr[:-m]))


def metricas(y, p, esc):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE": mape(y, p),
            "MASE": float(np.mean(np.abs(y - p)) / esc)}


def diebold_mariano(ea, eb, h=1, perda="quadratica"):
    """Teste de Diebold-Mariano (1995) com a correção de amostra pequena de
    Harvey, Leybourne e Newbold (1997).

    ea, eb : vetores de erro de previsão dos modelos A e B
    h      : horizonte (define a truncagem da variância de longo prazo)
    perda  : 'quadratica' (e²) ou 'absoluta' (|e|)

    H0: igual acurácia preditiva. DM < 0 favorece o modelo A.
    Devolve (estatística, p-valor bicaudal) sob t com n-1 g.l."""
    from scipy.stats import t as tdist
    ea, eb = np.asarray(ea, float), np.asarray(eb, float)
    if perda == "absoluta":
        d = np.abs(ea) - np.abs(eb)
    else:
        d = ea ** 2 - eb ** 2
    n = len(d); db = d.mean()
    g0 = np.sum((d - db) ** 2) / n
    g = [np.sum((d[k:] - db) * (d[:-k] - db)) / n for k in range(1, h)]
    var = (g0 + 2 * sum(g)) / n
    if var <= 0 or n < 3:
        return np.nan, np.nan
    dm = db / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)   # correção HLN
    return float(dm), float(2 * (1 - tdist.cdf(abs(dm), df=n - 1)))


def matriz_dm(erros, modelos, h=1, perda="quadratica"):
    """Matriz par-a-par de p-valores do DM. Célula (i, j) = p-valor de
    'modelo i tem a mesma acurácia que modelo j'."""
    M = pd.DataFrame(index=modelos, columns=modelos, dtype=float)
    for a in modelos:
        for b in modelos:
            if a == b:
                M.loc[a, b] = np.nan
            else:
                _, p = diebold_mariano(erros[a], erros[b], h=h, perda=perda)
                M.loc[a, b] = p
    return M


def forca_stl(serie, period=SAZONALIDADE):
    """Força de tendência (F_T) e de sazonalidade (F_S) de Hyndman &
    Athanasopoulos (2021, §3.3): 1 - Var(resíduo)/Var(resíduo + componente).
    Valores próximos de 1 indicam componente dominante."""
    st = STL(serie, period=period, robust=True).fit()
    r = np.asarray(st.resid); t = np.asarray(st.trend); s = np.asarray(st.seasonal)
    ft = max(0.0, 1 - np.var(r) / np.var(t + r))
    fs = max(0.0, 1 - np.var(r) / np.var(s + r))
    return float(ft), float(fs), st


# ==============================================================================
# 4) MODELOS
# ==============================================================================
def _feriados_mensais(idx):
    """Contagem de feriados nacionais brasileiros em DIA ÚTIL por mês.

    Justificativa: em frequência mensal, o add_country_holidays() nativo do
    Prophet só produz efeito quando o feriado cai exatamente no timestamp da
    observação (dia 1º), o que torna o termo quase inerte e cria um indicador
    espúrio de 'janeiro e maio'. A contagem mensal captura o que de fato
    importa para o volume de tráfego: quantos dias úteis o mês perdeu."""
    anos = sorted({d.year for d in idx})
    fer = pyholidays.Brazil(years=anos)
    out = []
    for d in idx:
        ini = pd.Timestamp(d.year, d.month, 1)
        fim = ini + pd.offsets.MonthEnd(0)
        dias = pd.date_range(ini, fim, freq="D")
        out.append(sum(1 for x in dias
                       if x.date() in fer and x.weekday() < 5))
    return np.array(out, dtype=float)


def m_naive(tr, h):
    u = tr.values[-SAZONALIDADE:]
    return np.array([u[i % SAZONALIDADE] for i in range(h)])


def m_hw(tr, h):
    return np.asarray(
        ExponentialSmoothing(tr, trend="add", seasonal="add",
                             seasonal_periods=SAZONALIDADE,
                             initialization_method="estimated")
        .fit().forecast(h))


def ordem_sarima(tr):
    """Grade de busca sobre p, q, P, Q por menor AIC.

    d e D são FIXADOS EM 1 a priori — não são selecionados pelo teste de raiz
    unitária. A justificativa é estrutural: a série exibe tendência de longo
    prazo (=> uma diferença simples) e sazonalidade anual determinística
    (=> uma diferença sazonal). Deixar d e D livres na grade produziria
    modelos não comparáveis entre si pelo AIC, que não é comparável entre
    ordens de diferenciação distintas."""
    best = (np.inf, None); d = D = 1
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        mm = SARIMAX(tr, order=(p, d, q),
                                     seasonal_order=(P, D, Q, SAZONALIDADE),
                                     enforce_stationarity=False,
                                     enforce_invertibility=False).fit(disp=False)
                        if mm.aic < best[0]:
                            best = (mm.aic, ((p, d, q), (P, D, Q, SAZONALIDADE)))
                    except Exception:
                        pass
    return best[1], best[0]


def descrever_ordem_sarima(ordem, aic):
    (p, d, q), (P, D, Q, s) = ordem
    return (f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]  |  AIC = {aic:.1f}\n"
            f"        parte não-sazonal: p={p} (AR), d={d} (diferenças), "
            f"q={q} (MA)\n"
            f"        parte sazonal:     P={P}, D={D}, Q={Q} a cada s={s} "
            f"meses\n"
            f"        (p, q, P e Q escolhidos por menor AIC; d e D fixados "
            f"em 1 a priori)")


def m_sarima(tr, h, ordem):
    o, so = ordem
    return np.asarray(
        SARIMAX(tr, order=o, seasonal_order=so, enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False).forecast(h))


def m_prophet(tr, h, modo=None):
    modo = PROPHET_FERIADOS if modo is None else modo
    df = pd.DataFrame({"ds": tr.index, "y": tr.values})
    mm = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                 daily_seasonality=False)
    if modo == "nativo":
        mm.add_country_holidays(country_name="BR")
    elif modo == "mensal":
        df["feriados"] = _feriados_mensais(tr.index)
        mm.add_regressor("feriados")
    mm.fit(df)
    fut = mm.make_future_dataframe(periods=h, freq="MS")
    if modo == "mensal":
        fut["feriados"] = _feriados_mensais(pd.DatetimeIndex(fut["ds"]))
    return mm.predict(fut)["yhat"].values[-h:]


def _feats(s):
    df = pd.DataFrame({"y": s})
    for l in (1, 2, 3, 12):
        df[f"lag_{l}"] = df["y"].shift(l)
    df["mm3"] = df["y"].shift(1).rolling(3).mean()
    df["mes"] = df.index.month
    return df


def m_xgb(tr, h):
    """Regressão supervisionada com defasagens; previsão multipasso recursiva
    (cada previsão realimenta a janela de defasagens do passo seguinte)."""
    df = _feats(tr).dropna()
    X = [c for c in df.columns if c != "y"]
    mm = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9,
                      random_state=SEED).fit(df[X], df["y"])
    hist = tr.copy(); out = []
    for _ in range(h):
        v = hist.values; nd = hist.index[-1] + pd.offsets.MonthBegin(1)
        row = {"lag_1": v[-1], "lag_2": v[-2], "lag_3": v[-3],
               "lag_12": v[-12] if len(v) >= 12 else v[0],
               "mm3": v[-3:].mean(), "mes": nd.month}
        yh = float(mm.predict(pd.DataFrame([row])[X])[0]); out.append(yh)
        hist = pd.concat([hist, pd.Series([yh], index=[nd])])
    return np.array(out)


def construir_modelos(ordem):
    """Devolve o dicionário nome -> função(treino, h)."""
    return {NAIVE:   lambda tr, h: m_naive(tr, h),
            HW:      lambda tr, h: m_hw(tr, h),
            SARIMA:  lambda tr, h: m_sarima(tr, h, ordem),
            PROPHET: lambda tr, h: m_prophet(tr, h),
            XGB:     lambda tr, h: m_xgb(tr, h)}


# ==============================================================================
# 5) BACKTEST POR ORIGEM MÓVEL
# ==============================================================================
def backtest_h1(serie, modelos, n_teste=N_TESTE):
    """Origem móvel com janela expansiva, um passo à frente.
    A cada origem t, cada modelo é reestimado com serie[:t] e prevê serie[t]."""
    n = len(serie); ini = n - n_teste
    prev = {k: [] for k in modelos}; reais = []; datas = []
    for t in range(ini, n):
        tr = serie.iloc[:t]
        reais.append(serie.iloc[t]); datas.append(serie.index[t])
        for nome, f in modelos.items():
            try:
                prev[nome].append(f(tr, 1)[0])
            except Exception:
                prev[nome].append(np.nan)
    dfp = pd.DataFrame(prev, index=datas); dfp["REAL"] = reais
    erros = {k: dfp["REAL"].values - dfp[k].values for k in modelos}
    return dfp, erros


def backtest_h(serie, modelos, h, n_teste=N_TESTE):
    """Origem móvel com janela expansiva e horizonte h.

    A cada origem, o modelo prevê h passos à frente e todos os h erros são
    guardados com o respectivo passo. Devolve um DataFrame longo com colunas
    origem / passo / data / real / modelo / previsto / erro."""
    n = len(serie); ini = n - n_teste
    linhas = []
    for t in range(ini, n - h + 1):
        tr = serie.iloc[:t]
        alvo = serie.iloc[t:t + h]
        for nome, f in modelos.items():
            try:
                yh = f(tr, h)
            except Exception:
                yh = np.full(h, np.nan)
            for k in range(h):
                linhas.append({"origem": serie.index[t - 1], "passo": k + 1,
                               "data": alvo.index[k], "real": float(alvo.iloc[k]),
                               "modelo": nome, "previsto": float(yh[k]),
                               "erro": float(alvo.iloc[k] - yh[k])})
    return pd.DataFrame(linhas)


def tabela_metricas(dfp, modelos, esc):
    tab = pd.DataFrame([{**metricas(dfp["REAL"], dfp[k], esc), "Modelo": k}
                        for k in modelos]).set_index("Modelo")
    return tab[["MAE", "RMSE", "MAPE", "MASE"]].sort_values("MASE")


# ==============================================================================
# 6) PROTOCOLO COMPLETO SOBRE UMA SÉRIE (reutilizado nos cenários)
# ==============================================================================
def rodar_protocolo(serie, rotulo, com_h12=False):
    """Executa ordem do SARIMA, backtest h=1 (e h=12), métricas, DM e
    Ljung-Box sobre uma série. Devolve um dicionário de resultados."""
    n = len(serie); ini = n - N_TESTE
    treino = serie.iloc[:ini]
    esc = escala_mase(treino)

    diga(f"\n  [{rotulo}] selecionando a ordem do SARIMA por AIC...")
    ordem, aic = ordem_sarima(treino)
    diga("    " + descrever_ordem_sarima(ordem, aic))

    modelos = construir_modelos(ordem)
    diga(f"  [{rotulo}] backtest h = 1 ({N_TESTE} origens)...")
    dfp, erros = backtest_h1(serie, modelos)
    tab1 = tabela_metricas(dfp, modelos, esc)

    res = {"rotulo": rotulo, "serie": serie, "esc": esc, "ordem": ordem,
           "aic": aic, "modelos": modelos, "dfp": dfp, "erros": erros,
           "tab_h1": tab1, "treino": treino}

    # --- Diebold-Mariano: matriz completa, duas perdas ------------------------
    nomes = [m for m in ORDEM_EXIBICAO if m in modelos]
    res["dm_quad"] = matriz_dm(erros, nomes, h=1, perda="quadratica")
    res["dm_abs"]  = matriz_dm(erros, nomes, h=1, perda="absoluta")

    # --- Ljung-Box e viés sobre os erros de previsão do melhor modelo ---------
    melhor = tab1.index[0]
    e = pd.Series(erros[melhor]).dropna()
    lb = {}
    for L in (6, 12):
        if len(e) > L:
            lb[L] = float(acorr_ljungbox(e, lags=[L],
                                         return_df=True)["lb_pvalue"].iloc[0])
    from scipy.stats import ttest_1samp
    t_v, p_v = ttest_1samp(e, 0.0)
    res.update({"melhor": melhor, "ljung_box": lb,
                "vies_media": float(e.mean()), "vies_t": float(t_v),
                "vies_p": float(p_v)})

    # --- Backtest h = 12 ------------------------------------------------------
    if com_h12:
        diga(f"  [{rotulo}] backtest h = 12 ({N_TESTE - 12 + 1} origens, "
             f"cada uma prevendo 12 meses)...")
        longo = backtest_h(serie, modelos, 12)
        agg = (longo.groupby("modelo")
                    .apply(lambda g: pd.Series({
                        "MAE": np.mean(np.abs(g["erro"])),
                        "RMSE": np.sqrt(np.mean(g["erro"] ** 2)),
                        "MAPE": np.mean(np.abs(g["erro"] / g["real"])) * 100,
                        "MASE": np.mean(np.abs(g["erro"])) / esc}))
                    .sort_values("MASE"))
        por_passo = (longo.assign(abse=lambda d: np.abs(d["erro"]) / esc)
                          .pivot_table(index="passo", columns="modelo",
                                       values="abse", aggfunc="mean"))
        res["tab_h12"] = agg
        res["mase_por_passo"] = por_passo
        res["longo_h12"] = longo
    return res


# ==============================================================================
# 7) FIGURAS
# ==============================================================================
def _fmt_milhoes(x, pos):
    return f"{x/1e6:.2f}".replace(".", ",")


def _estilo(ax, ylab="Volume mensal (milhões de veículos)"):
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_milhoes))
    ax.set_ylabel(ylab); ax.grid(alpha=.3, linewidth=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_stl(serie, praca, arq="fig1_stl.png"):
    st = STL(serie, period=SAZONALIDADE, robust=True).fit()
    fg, axs = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
    for a, (dados, tit, cor) in zip(axs, [
            (serie.values, "Observado", "black"),
            (st.trend, "Tendência", "#1f77b4"),
            (st.seasonal, "Sazonal", "#2ca02c"),
            (st.resid, "Resíduo", "#d62728")]):
        a.plot(serie.index, dados, color=cor, lw=1.2)
        a.set_ylabel(tit, fontsize=9); a.grid(alpha=.3, linewidth=.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axs[0].set_title(f"Decomposição STL — praça de {praca}", fontsize=11)
    axs[-1].set_xlabel("Ano")
    fg.tight_layout(); fg.savefig(arq, dpi=300); plt.show()


def fig_backtest(serie, dfp, modelos, arq="fig2_backtest.png"):
    corte = serie.index[len(serie) - N_TESTE]
    fg, ax = plt.subplots(figsize=(12, 6))
    ax.plot(serie.index, serie.values, color="black", lw=1.3, label="Série real")
    ax.axvspan(corte, serie.index[-1], color="orange", alpha=.08)
    ax.axvline(corte, color="gray", ls=":", lw=1)
    for nome in modelos:
        ax.plot(dfp.index, dfp[nome], lw=1.1, label=nome)
    ax.set_title("Separação treino/teste e desempenho no período de teste "
                 "(origem móvel, h = 1)", fontsize=11)
    ax.set_xlabel("Ano"); _estilo(ax)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fg.tight_layout(); fg.savefig(arq, dpi=300); plt.show()


def fig_mase(tab, arq="fig3_mase.png", titulo="Acurácia relativa (MASE, h = 1)"):
    fg, ax = plt.subplots(figsize=(9, 4.5))
    od = tab.sort_values("MASE")
    cores = ["#1f77b4" if m == od.index[0] else "#a9c7e0" for m in od.index]
    ax.barh(od.index, od["MASE"], color=cores)
    ax.axvline(1.0, color="red", ls="--", lw=1, label="MASE = 1 (naïve sazonal)")
    for i, v in enumerate(od["MASE"]):
        ax.text(v + .008, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=9)
    ax.set_xlabel("MASE (menor é melhor)"); ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3, axis="x", linewidth=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fg.tight_layout(); fg.savefig(arq, dpi=300); plt.show()


def fig_futuro(serie, fut_idx, futd, praca, arq="fig4_previsao.png"):
    fg, ax = plt.subplots(figsize=(13, 6))
    ax.plot(serie.index, serie.values, color="black", lw=1.4,
            label="Histórico observado")
    for nome, yv in futd.items():
        ax.plot(fut_idx, yv, "--", lw=1.4, label=nome)
    ax.axvspan(pd.Timestamp(COVID_INICIO), pd.Timestamp(COVID_FIM),
               color="red", alpha=.07)
    ax.set_title(f"Volume mensal de tráfego e previsões — praça de {praca}",
                 fontsize=11)
    ax.set_xlabel("Ano"); _estilo(ax)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fg.tight_layout(); fg.savefig(arq, dpi=300); plt.show()


def fig_por_passo(por_passo, arq="fig5_mase_por_horizonte.png"):
    fg, ax = plt.subplots(figsize=(9, 5))
    for nome in por_passo.columns:
        ax.plot(por_passo.index, por_passo[nome], marker="o", ms=3.5,
                lw=1.3, label=nome)
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.set_xlabel("Passos à frente (meses)"); ax.set_ylabel("MASE médio")
    ax.set_title("Degradação da acurácia com o horizonte de previsão", fontsize=11)
    ax.set_xticks(range(1, len(por_passo) + 1))
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=.3, linewidth=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fg.tight_layout(); fg.savefig(arq, dpi=300); plt.show()


# ==============================================================================
# 8) EXPORTAÇÃO PARA EXCEL
# ==============================================================================
def _estilizar(ws):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    HDR = Font(name="Arial", bold=True, color="FFFFFF")
    FILL = PatternFill("solid", fgColor="305496")
    CEN = Alignment(horizontal="center")
    TH = Border(*[Side(style="thin", color="D9D9D9")] * 4)
    body = Font(name="Arial")
    for c in ws[1]:
        c.font = HDR; c.fill = FILL; c.alignment = CEN
    ws.freeze_panes = "A2"
    for col in ws.columns:
        w = max((len(str(c.value)) if c.value is not None else 0) for c in col) + 3
        ws.column_dimensions[col[0].column_letter].width = max(11, w)
        for c in col:
            if c.row > 1:
                c.font = body; c.border = TH


def exportar_excel(res, futd, fut_idx, praca, extras=None, arq="serie_prevista.xlsx"):
    from openpyxl import Workbook
    from openpyxl.worksheet.formula import ArrayFormula
    serie, dfp, tab = res["serie"], res["dfp"], res["tab_h1"]
    modelos = list(res["modelos"].keys())
    wb = Workbook()

    ws = wb.active; ws.title = "Serie_Historica"
    ws.append(["Data", "Volume", "Particao"])
    for i, (d, v) in enumerate(serie.items()):
        ws.append([d.strftime("%Y-%m"), round(float(v), 2),
                   "Treino" if i < len(serie) - N_TESTE else "Teste"])
    _estilizar(ws)

    ws2 = wb.create_sheet("Backtest_h1"); ws2.append(["Data", "Real"] + modelos)
    for d, row in dfp.iterrows():
        ws2.append([d.strftime("%Y-%m"), round(float(row["REAL"]), 2)] +
                   [round(float(row[m]), 2) for m in modelos])
    _estilizar(ws2)

    ws3 = wb.create_sheet("Previsao_Futura"); ws3.append(["Data"] + list(futd))
    for j, d in enumerate(fut_idx):
        ws3.append([d.strftime("%Y-%m")] + [round(float(futd[m][j]), 2)
                                            for m in futd])
    _estilizar(ws3)

    # Métricas por fórmula (conferíveis na própria planilha)
    ws4 = wb.create_sheet("Metricas_h1")
    ws4.append(["Modelo", "MAE", "RMSE", "MAPE (%)", "MASE"])
    n = len(dfp); r0, r1 = 2, 1 + n
    esc = res["esc"]
    cmap = {modelos[i]: chr(ord("C") + i) for i in range(len(modelos))}
    ri = 1
    for m in modelos:
        col = cmap[m]; ri += 1
        R = f"Backtest_h1!B{r0}:B{r1}"; C = f"Backtest_h1!{col}{r0}:{col}{r1}"
        ws4.cell(row=ri, column=1, value=m)
        ws4.cell(row=ri, column=2).value = ArrayFormula(f"B{ri}", f"=AVERAGE(ABS({R}-{C}))")
        ws4.cell(row=ri, column=3).value = ArrayFormula(f"C{ri}", f"=SQRT(AVERAGE(({R}-{C})^2))")
        ws4.cell(row=ri, column=4).value = ArrayFormula(f"D{ri}", f"=AVERAGE(ABS(({R}-{C})/{R}))*100")
        ws4.cell(row=ri, column=5).value = f"=B{ri}/{esc:.6f}"
    _estilizar(ws4)
    ws4.append([]); ws4.append([f"Praça: {praca}"])
    ws4.append([f"Escala MASE (erro naïve sazonal no treino) = {esc:.2f}"])

    for nome, M in (("DM_quadratica", res["dm_quad"]), ("DM_absoluta", res["dm_abs"])):
        wsx = wb.create_sheet(nome)
        wsx.append(["Modelo"] + list(M.columns))
        for idx, row in M.iterrows():
            wsx.append([idx] + [None if pd.isna(v) else round(float(v), 4)
                                for v in row])
        _estilizar(wsx)

    if "tab_h12" in res:
        wsh = wb.create_sheet("Metricas_h12")
        wsh.append(["Modelo", "MAE", "RMSE", "MAPE (%)", "MASE"])
        for idx, row in res["tab_h12"].iterrows():
            wsh.append([idx] + [round(float(v), 4) for v in row])
        _estilizar(wsh)
        wsp = wb.create_sheet("MASE_por_passo")
        wsp.append(["Passo"] + list(res["mase_por_passo"].columns))
        for idx, row in res["mase_por_passo"].iterrows():
            wsp.append([int(idx)] + [round(float(v), 4) for v in row])
        _estilizar(wsp)

    for nome, tabx in (extras or {}).items():
        wse = wb.create_sheet(nome[:31])
        wse.append(["Modelo"] + list(tabx.columns))
        for idx, row in tabx.iterrows():
            wse.append([idx] + [round(float(v), 4) for v in row])
        _estilizar(wse)

    wb.save(arq)


# ==============================================================================
# 9) MAIN
# ==============================================================================
def main():
    diga("=" * 78)
    diga(" PREVISÃO DE TRÁFEGO EM RODOVIAS CONCEDIDAS — DADOS ABERTOS DA ANTT")
    diga(" Versão final do TCC")
    diga("=" * 78)

    # ---------------------------------------------------------------- dados --
    prontas = None
    if ARQUIVO_SERIES and os.path.exists(ARQUIVO_SERIES):
        diga(f"\n[1] Lendo séries já extraídas de {ARQUIVO_SERIES}")
        prontas = carregar_series_prontas(ARQUIVO_SERIES)
        base = None
        diga(f"    classes disponíveis: {', '.join(prontas)}")
    arqs = [] if prontas is not None else localizar_csvs()
    if prontas is None and not arqs:
        diga("\n[!] Nenhum CSV encontrado. Tentando baixar...")
        try:
            baixar_dados(); arqs = localizar_csvs()
        except Exception as e:
            diga(f"    Falha no download: {e}")
            diga("    -> Libere 'dados.antt.gov.br' no egress OU coloque os "
                 "CSVs em ./dados")
            return
    if prontas is None:
        diga(f"\n[1] Lendo {len(arqs)} arquivo(s)...")
        base = carregar_tudo(arqs)
        diga(f"    período coberto: {base['data'].min():%Y-%m} a "
             f"{base['data'].max():%Y-%m}")

    diga("\n[2] Construindo a série da praça-alvo...")
    if prontas is not None:
        serie, praca, diag = serie_de_prontas(prontas, "total")
    else:
        serie, praca, diag = construir_serie(base, "total")
    if ANONIMIZAR:
        diga("    (concessionária omitida por opção de anonimização — "
             "ANONIMIZAR = True)")
    else:
        diga(f"    concessionária: {CONCESSIONARIA_ALVO}")

    diga("\n[2b] Estatística descritiva da série:")
    rot = {"count": "nº de meses", "mean": "média", "std": "desvio-padrão",
           "min": "mínimo", "25%": "1º quartil", "50%": "mediana",
           "75%": "3º quartil", "max": "máximo"}
    for k, v in serie.describe().items():
        dec = 0 if k != "count" else 0
        diga(f"    {rot.get(k, k):>22}: {br(v, dec):>16}")
    tri_ini = serie.iloc[:36].mean(); tri_fim = serie.iloc[-36:].mean()
    diga(f"    {'média do 1º triênio':>22}: {br(tri_ini):>16}")
    diga(f"    {'média do último triênio':>22}: {br(tri_fim):>16}")
    diga(f"    {'crescimento entre eles':>22}: "
         f"{br(100*(tri_fim/tri_ini-1), 1):>16} %")
    piso = serie.loc[pd.Timestamp(COVID_INICIO):pd.Timestamp(COVID_FIM)]
    if len(piso):
        mes_piso = piso.idxmin(); v_piso = float(piso.min())
        jan = serie.get(pd.Timestamp(f"{mes_piso.year}-01-01"), np.nan)
        txt = f"    piso pandêmico: {mes_piso:%Y-%m} = {br(v_piso)}"
        if jan == jan:
            txt += f"  ({br(100*(v_piso/jan-1), 1)} % ante janeiro de {mes_piso.year})"
        diga(txt)

    # ------------------------------------------------------------ diagnóstico -
    diga("\n[3] Diagnóstico da série")
    ft, fs, _ = forca_stl(serie)
    diga(f"    - Força da tendência (F_T)     = {br(ft, 3)}")
    diga(f"    - Força da sazonalidade (F_S)  = {br(fs, 3)}")
    diga("      (Hyndman & Athanasopoulos, 2021: próximo de 1 = componente "
         "dominante)")
    adf_stat, adf_p = adfuller(serie.dropna(), regression="ct", autolag="AIC")[:2]
    diga(f"    - ADF  (H0: há raiz unitária): estatística = {br(adf_stat, 3)}, "
         f"p = {br(adf_p, 3)}")
    try:
        kp_stat, kp_p = kpss(serie.dropna(), regression="ct", nlags="auto")[:2]
        diga(f"    - KPSS (H0: é estacionária) : estatística = {br(kp_stat, 3)}, "
             f"p = {br(kp_p, 3)}")
    except Exception as e:
        diga(f"    - KPSS não pôde ser calculado: {e}")
    diga("      Leitura conjunta: os dois testes têm hipóteses nulas opostas; "
         "reportá-los")
    diga("      juntos evita concluir estacionariedade por simples não "
         "rejeição do ADF.")
    diga("      Observação: d = 1 e D = 1 no SARIMA foram FIXADOS a priori "
         "(tendência +")
    diga("      sazonalidade anual), e não selecionados por estes testes — o "
         "AIC não é")
    diga("      comparável entre ordens de diferenciação distintas.")

    # ------------------------------------------------- cenário A: série original
    diga("\n" + "-" * 78)
    diga("[4] CENÁRIO A — série original (quebra pandêmica não tratada)")
    diga("-" * 78)
    resA = rodar_protocolo(serie, "A", com_h12=RODAR_H12)

    diga("\n  Métricas — backtest h = 1 (menor é melhor):")
    diga(textwrap.indent(tabela_br(resA["tab_h1"]), "    "))
    diga(f"    escala do MASE (erro naïve sazonal no treino) = "
         f"{br(resA['esc'])}")
    diga("    Nota: o denominador do MASE é calculado no TREINO, que contém a "
         "pandemia;")
    diga("    a janela de teste é um período calmo. É por isso que o próprio "
         "naïve sazonal")
    diga("    pode exibir MASE ligeiramente inferior a 1 no teste.")

    if RODAR_H12:
        diga("\n  Métricas — backtest h = 12 (pool de todos os passos):")
        diga(textwrap.indent(tabela_br(resA["tab_h12"]), "    "))
        diga("\n  MASE médio por passo à frente:")
        diga(textwrap.indent(tabela_br(resA["mase_por_passo"]), "    "))
        diga("    Nota: não se reporta teste de Diebold-Mariano para h = 12. "
             "As trajetórias")
        diga("    multipasso de origens sucessivas se sobrepõem, e os erros "
             "empilhados não")
        diga("    formam uma série que satisfaça as condições da estatística "
             "DM padrão.")

    diga("\n  Diebold-Mariano — p-valores par-a-par (h = 1, perda QUADRÁTICA):")
    diga(textwrap.indent(tabela_br(resA["dm_quad"].astype(float)), "    "))
    diga("\n  Diebold-Mariano — p-valores par-a-par (h = 1, perda ABSOLUTA):")
    diga(textwrap.indent(tabela_br(resA["dm_abs"].astype(float)), "    "))
    diga("    Leitura: p < 0,05 => diferença de acurácia estatisticamente "
         "significativa.")
    diga("    Reporta-se sob as duas perdas porque o ranking é feito por MASE "
         "(erro")
    diga("    absoluto) e o DM clássico usa perda quadrática; a conclusão só é "
         "robusta")
    diga("    se resistir às duas.")

    melhor = resA["melhor"]
    diga(f"\n  Diagnóstico dos erros de previsão de {melhor} "
         f"(h = 1, n = {N_TESTE}):")
    for L, p in resA["ljung_box"].items():
        diga(f"    - Ljung-Box com {L} defasagens: p = {br(p, 3)}")
    diga(f"    - Viés: erro médio = {br(resA['vies_media'])}; "
         f"t = {br(resA['vies_t'], 3)}; p = {br(resA['vies_p'], 3)}")
    diga(f"    Advertência de poder: com n = {N_TESTE}, o Ljung-Box a 12 "
         "defasagens tem")
    diga("    poder baixo; a não rejeição é evidência fraca, não prova de "
         "adequação.")
    diga("    Observe ainda que o teste incide sobre os ERROS DE PREVISÃO "
         "FORA DA AMOSTRA")
    diga("    (um passo à frente), e não sobre resíduos de ajuste dentro da "
         "amostra.")

    # ------------------------------------- cenário B: sensibilidade à pandemia
    resB = None
    if RODAR_SENSIBILIDADE_COVID:
        diga("\n" + "-" * 78)
        diga("[5] CENÁRIO B — sensibilidade: quebra pandêmica ajustada")
        diga("-" * 78)
        serie_aj, meses_aj = ajustar_quebra_covid(serie)
        diga(f"    meses ajustados ({len(meses_aj)}): "
             f"{', '.join(d.strftime('%Y-%m') for d in meses_aj)}")
        diga("    método: substituição pelo sinal 'tendência + sazonal' de um "
             "STL robusto")
        diga("    estimado SOMENTE no trecho de treino. O mesmo ajuste é "
             "aplicado a todos")
        diga("    os modelos, preservando a equidade da comparação.")
        resB = rodar_protocolo(serie_aj, "B", com_h12=False)
        diga("\n  Métricas — cenário B, h = 1:")
        diga(textwrap.indent(tabela_br(resB["tab_h1"]), "    "))
        diga("\n  Comparação entre cenários (posição por MASE e MAPE):")
        cmp = pd.DataFrame({
            "Posição A": {m: i + 1 for i, m in
                          enumerate(resA["tab_h1"].index)},
            "Posição B": {m: i + 1 for i, m in
                          enumerate(resB["tab_h1"].index)},
            "MAPE A (%)": resA["tab_h1"]["MAPE"].round(3),
            "MAPE B (%)": resB["tab_h1"]["MAPE"].round(3)})
        diga(textwrap.indent(tabela_br(cmp), "    "))
        estavel = list(resA["tab_h1"].index) == list(resB["tab_h1"].index)
        diga(f"    -> ranking {'ESTÁVEL' if estavel else 'ALTERADO'} entre os "
             "cenários.")
        diga("    ATENÇÃO ao transcrever: o MASE NÃO é comparável entre A e B. "
             "Seu denominador")
        diga("    é o erro do naïve sazonal no treino de cada cenário, e o "
             "treino ajustado não")
        diga("    contém o choque de 2020 — o denominador encolhe e o MASE do "
             "cenário B sobe")
        diga("    mecanicamente. Compare os cenários pelo RANKING e pelo MAPE, "
             "nunca pelo MASE.")

    # --------------------------------------------- comparação de feriados -----
    if RODAR_COMPARACAO_FERIADOS:
        diga("\n" + "-" * 78)
        diga("[6] Prophet — sensibilidade ao tratamento de feriados (h = 1)")
        diga("-" * 78)
        linhas = []
        for modo in ("nenhum", "nativo", "mensal"):
            prev, reais = [], []
            for t in range(len(serie) - N_TESTE, len(serie)):
                tr = serie.iloc[:t]
                try:
                    prev.append(m_prophet(tr, 1, modo=modo)[0])
                except Exception:
                    prev.append(np.nan)
                reais.append(serie.iloc[t])
            linhas.append({"Tratamento": modo,
                           **metricas(reais, prev, resA["esc"])})
        tab_f = pd.DataFrame(linhas).set_index("Tratamento")
        diga(textwrap.indent(tabela_br(tab_f), "    "))
        diga("    O modo 'nativo' (add_country_holidays) é praticamente inerte "
             "em base")
        diga("    mensal: só produz efeito quando o feriado cai no dia 1º, o "
             "que gera um")
        diga("    indicador espúrio de janeiro e maio. O modo 'mensal' usa a "
             "contagem de")
        diga("    feriados nacionais em dia útil no mês — o que de fato "
             "desloca o volume.")
        diga(f"    Modo adotado no trabalho: '{PROPHET_FERIADOS}'.")

    # ------------------------------------------------------------- segmentos --
    extras = {}
    if RODAR_SEGMENTOS:
        diga("\n" + "-" * 78)
        diga("[7] Segmentos de veículo (h = 1)")
        diga("-" * 78)
        for classe in ("leves", "pesados"):
            try:
                if prontas is not None:
                    s_c, _, dg = serie_de_prontas(prontas, classe,
                                                  verboso=False)
                else:
                    s_c, _, dg = construir_serie(base, classe, verboso=False)
                diga(f"\n  Segmento {classe.upper()} — {len(s_c)} meses; "
                     f"{dg['n_faltantes']} ausente(s).")
                ft_c, fs_c, _ = forca_stl(s_c)
                diga(f"    F_T = {br(ft_c, 3)} | F_S = {br(fs_c, 3)}")
                r_c = rodar_protocolo(s_c, classe, com_h12=False)
                diga(textwrap.indent(tabela_br(r_c["tab_h1"]), "    "))
                extras[f"Metricas_{classe}"] = r_c["tab_h1"]
                diga(f"    melhor modelo em {classe}: {r_c['tab_h1'].index[0]}")
            except Exception as e:
                diga(f"    [!] Segmento {classe} não pôde ser processado: {e}")

    # -------------------------------------------------- previsão futura + figs
    diga("\n" + "-" * 78)
    diga(f"[8] Previsão futura ({HORIZONTE_FUT} meses), figuras e planilha")
    diga("-" * 78)
    ordem_full, aic_full = ordem_sarima(serie)
    diga("    ordem do SARIMA reestimada na série completa:")
    diga("    " + descrever_ordem_sarima(ordem_full, aic_full))
    modelos_full = construir_modelos(ordem_full)
    fut_idx = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1),
                            periods=HORIZONTE_FUT, freq="MS")
    futd = {}
    for nome, f in modelos_full.items():
        try:
            futd[nome] = f(serie, HORIZONTE_FUT)
        except Exception as e:
            diga(f"    [!] {nome} falhou na previsão futura: {e}")

    fig_stl(serie, praca)
    fig_backtest(serie, resA["dfp"], resA["modelos"])
    fig_mase(resA["tab_h1"])
    fig_futuro(serie, fut_idx, futd, praca)
    if RODAR_H12:
        fig_por_passo(resA["mase_por_passo"])

    exportar_excel(resA, futd, fut_idx, praca, extras=extras)
    resA["tab_h1"].round(4).to_csv("metricas_h1.csv")
    if RODAR_H12:
        resA["tab_h12"].round(4).to_csv("metricas_h12.csv")
    serie.to_csv("serie_praca.csv")
    diga("    gravados: fig1_stl.png, fig2_backtest.png, fig3_mase.png,")
    diga("              fig4_previsao.png" +
         (", fig5_mase_por_horizonte.png" if RODAR_H12 else "") + ",")
    diga("              serie_prevista.xlsx, metricas_h1.csv, serie_praca.csv")

    # ------------------------------------------- bloco para transcrição -------
    diga("\n" + "=" * 78)
    diga(" RESULTADOS PARA TRANSCRIÇÃO NO TCC")
    diga(" (cada item indica a seção do texto em que entra)")
    diga("=" * 78)
    t1 = resA["tab_h1"]
    diga(f"\n [Metodologia > Fonte de Dados]")
    diga(f"   Meses da série ..................: {diag['n_meses']}")
    diga(f"   Observações ausentes ............: {diag['n_faltantes']}"
         + (f" ({', '.join(diag['meses_faltantes'])})"
            if diag["n_faltantes"] else ""))
    diga(f"   Ausentes na janela de teste .....: "
         f"{len(diag['faltantes_no_teste'])}")
    diga("\n [Metodologia > Modelos Avaliados]")
    diga(f"   Ordem do SARIMA — estimada no TREINO .....: "
         f"{fmt_ordem(resA['ordem'])}  (AIC = {br(resA['aic'], 1)})")
    diga(f"   Ordem do SARIMA — série completa .........: "
         f"{fmt_ordem(ordem_full)}  (AIC = {br(aic_full, 1)})")
    diga("   (a primeira é a usada no backtest; a segunda, na previsão futura)")
    diga("\n [Resultados > Caracterização e Decomposição]")
    diga(f"   Força da tendência F_T ...................: {br(ft, 3)}")
    diga(f"   Força da sazonalidade F_S ................: {br(fs, 3)}")
    diga(f"   ADF: estatística = {br(adf_stat, 3)}, p = {br(adf_p, 3)}")
    diga("\n [Resultados > Desempenho Comparativo] — h = 1")
    for m in t1.index:
        r = t1.loc[m]
        diga(f"   {m:<16}: MAE {br(r['MAE']):>10} | RMSE {br(r['RMSE']):>10} "
             f"| MAPE {br(r['MAPE'], 2):>5} % | MASE {br(r['MASE'], 3)}")
    if RODAR_H12:
        diga("\n [Resultados > Desempenho Comparativo] — h = 12")
        for m in resA["tab_h12"].index:
            r = resA["tab_h12"].loc[m]
            diga(f"   {m:<16}: MAE {br(r['MAE']):>10} | RMSE {br(r['RMSE']):>10} "
                 f"| MAPE {br(r['MAPE'], 2):>5} % | MASE {br(r['MASE'], 3)}")
    diga("\n [Resultados > Significância das Diferenças]")
    nomes = list(t1.index)
    campeao = nomes[0]
    for outro in nomes[1:]:
        pq = float(resA["dm_quad"].loc[campeao, outro])
        pa = float(resA["dm_abs"].loc[campeao, outro])
        if pq < .05 and pa < .05:
            vq = "significativa sob as duas perdas"
        elif pq < .05 or pa < .05:
            vq = "significativa apenas sob uma das perdas — NÃO afirme superioridade"
        else:
            vq = "não significativa"
        diga(f"   {campeao} x {outro:<16}: p(quadrática) = {br(pq, 3)} | "
             f"p(absoluta) = {br(pa, 3)}  ->  {vq}")
    diga("\n [Resultados > Diagnóstico de Resíduos]")
    for L, p in resA["ljung_box"].items():
        diga(f"   Ljung-Box ({L} defasagens) de {campeao}: p = {br(p, 3)}")
    diga(f"   Viés (erro médio) ........................: "
         f"{br(resA['vies_media'])}; p = {br(resA['vies_p'], 3)}")
    if resB is not None:
        diga("\n [Resultados > Sensibilidade à Quebra Estrutural]")
        diga(f"   Ranking cenário A (original): "
             f"{' > '.join(resA['tab_h1'].index)}")
        diga(f"   Ranking cenário B (ajustada): "
             f"{' > '.join(resB['tab_h1'].index)}")
        diga(f"   MAPE do 1º colocado: A = "
             f"{br(resA['tab_h1']['MAPE'].iloc[0], 2)} % | B = "
             f"{br(resB['tab_h1']['MAPE'].iloc[0], 2)} %")
        diga("   (compare por MAPE, não por MASE: os denominadores do MASE "
             "diferem entre cenários)")
        estavel = list(resA["tab_h1"].index) == list(resB["tab_h1"].index)
        diga(f"   Ranking {'estável' if estavel else 'alterado'} entre cenários"
             " — esta é a frase que sustenta a alegação de sensibilidade no texto.")
    if extras:
        diga("\n [Resultados > Comportamento por Segmento]")
        for k, tv in extras.items():
            diga(f"   {k}: melhor = {tv.index[0]}, "
                 f"MASE = {br(tv['MASE'].iloc[0], 3)}, "
                 f"MAPE = {br(tv['MAPE'].iloc[0], 2)} %")
    diga("\n" + "=" * 78)

    with open("resultados_para_texto.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(_RELATORIO))
    print("\nArquivo 'resultados_para_texto.txt' gravado.")
    print("Concluído.")


# ==============================================================================
# 10) ANÁLISES COMPLEMENTARES DO TCC
# ==============================================================================
RODAR_PIB_ELASTICIDADE = False   # exercício retirado do TCC (integridade dos dados de 2025)
ARQ_2025   = os.path.join(PASTA_DADOS, "volume-trafego-praca-pedagio-2025_mensal_consolidado.csv")
PIB_FOCUS  = {2024: 0.0317, 2025: 0.0195}     # Focus de 22/11/2024 (medianas)
ELASTICIDADES = {"CNT (2026)": 1.61, "unitária": 1.00, "nula (= naïve sazonal)": 0.0}
MESES_EXCLUIDOS_2025 = (1, 9)                 # jan e set/2025: falhas de registro
TIR_BR381, PRAZO, CRESC_BASE = 0.1197, 30, 0.02


def _serie_principal():
    base = carregar_tudo(localizar_csvs())
    serie, _, _ = construir_serie(base, "total", verboso=False)
    return serie


def analises_complementares(serie):
    from statsmodels.stats.multitest import multipletests
    from pmdarima.arima import nsdiffs
    out = []
    p = lambda s="": (print(s), out.append(s))
    tr = serie.iloc[:-N_TESTE]
    escala = escala_mase(tr)

    p("=" * 78); p(" ANÁLISES COMPLEMENTARES"); p("=" * 78)

    # (a) Diebold-Mariano com correção de Holm
    ordem, _ = ordem_sarima(tr)
    modelos = construir_modelos(ordem)
    _, erros = backtest_h1(serie, modelos)
    nomes = [HW, SARIMA, PROPHET, XGB, NAIVE]
    pares = [(a, b) for i, a in enumerate(nomes) for b in nomes[i + 1:]]
    pq = [diebold_mariano(erros[a], erros[b], h=1, perda="quadratica")[1] for a, b in pares]
    pa = [diebold_mariano(erros[a], erros[b], h=1, perda="absoluta")[1] for a, b in pares]
    holm = multipletests(pq + pa, method="holm")[1]
    p("\n[a] Diebold-Mariano (h = 1) — p-valores brutos e ajustados por Holm (20 testes)")
    p(f"    {'par':34s} {'quad':>7s} {'Holm':>7s} {'abs':>7s} {'Holm':>7s}")
    for k, (a, b) in enumerate(pares):
        p(f"    {a + ' x ' + b:34s} {pq[k]:7.3f} {holm[k]:7.3f} {pa[k]:7.3f} {holm[10 + k]:7.3f}")

    # (b) raiz unitária sazonal
    p("\n[b] Raiz unitária sazonal (número de diferenças sazonais indicado)")
    for nome, x in (("série completa", serie), ("treino", tr)):
        p(f"    {nome:15s} OCSB: D = {nsdiffs(x.values, m=12, test='ocsb')}   "
          f"Canova-Hansen: D = {nsdiffs(x.values, m=12, test='ch')}")

    # (c) SARIMA com D = 0 (sensibilidade)
    best = (np.inf, None)
    for pp in range(3):
        for qq in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        r = SARIMAX(tr, order=(pp, 1, qq), seasonal_order=(P, 0, Q, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=False)
                        if r.aic < best[0]:
                            best = (r.aic, ((pp, 1, qq), (P, 0, Q, 12)))
                    except Exception:
                        pass
    o = best[1]
    f0 = lambda trn, h: np.asarray(SARIMAX(trn, order=o[0], seasonal_order=o[1],
                                           enforce_stationarity=False,
                                           enforce_invertibility=False).fit(disp=False).forecast(h))
    d1, _ = backtest_h1(serie, {"SARIMA D=0": f0})
    lg = backtest_h(serie, {"SARIMA D=0": f0}, 12)
    m1 = metricas(d1["REAL"].values, d1["SARIMA D=0"].values, escala)
    m12 = metricas(lg.real.values, lg.previsto.values, escala)
    p(f"\n[c] SARIMA com D = 0: ordem {o}; MASE h=1 = {m1['MASE']:.3f}; MASE h=12 = {m12['MASE']:.3f}")

    # (d) viés de 1 p.p. na taxa de crescimento
    t = np.arange(1, PRAZO + 1)
    base = ((1 + CRESC_BASE) ** t / (1 + TIR_BR381) ** t).sum()
    p(f"\n[d] Viés de 1 p.p. (prazo {PRAZO} anos, taxa {TIR_BR381:.2%}, crescimento-base {CRESC_BASE:.0%})")
    for b in (0.01, -0.01):
        pv = ((1 + CRESC_BASE + b) ** t / (1 + TIR_BR381) ** t).sum()
        p(f"    {b:+.0%}: tráfego no ano {PRAZO} {((1 + CRESC_BASE + b) / (1 + CRESC_BASE)) ** PRAZO - 1:+.1%}; "
          f"VP da receita {pv / base - 1:+.1%}; tarifa p/ VPL nulo {base / pv - 1:+.1%}")

    # (e) PIB x elasticidade (Focus de 22/11/2024) contra o realizado de 2025
    if RODAR_PIB_ELASTICIDADE and os.path.exists(ARQ_2025):
        d = pd.read_csv(ARQ_2025, sep=";", encoding="latin-1", dtype=str)
        d = d[d["praca"].str.contains(PRACA_ALVO, case=False, na=False)]
        d["v"] = pd.to_numeric(d["volume_total"].str.replace(".", "", regex=False)
                               .str.replace(",", ".", regex=False), errors="coerce")
        real = d.groupby("mes_ano")["v"].sum()
        real.index = pd.to_datetime(real.index, format="%m/%Y")
        fut = pd.read_excel("serie_prevista.xlsx", sheet_name="Previsao_Futura")
        fut.index = pd.to_datetime(fut["Data"])
        idx = [x for x in fut.index if x.year == 2025 and x.month not in MESES_EXCLUIDOS_2025]
        R = real.loc[idx]
        res = {f"PIB x E ({k}, {E})": pd.Series([serie[x - pd.DateOffset(years=1)] *
                                                  (1 + E * PIB_FOCUS[x.year]) for x in idx], index=idx)
               for k, E in ELASTICIDADES.items()}
        res.update({k: fut.loc[idx, k] for k in [HW, SARIMA, PROPHET, XGB]})
        p(f"\n[e] PIB x elasticidade vs. realizado em {len(idx)} meses de 2025 "
          f"(excluídos dez/2024 e meses {MESES_EXCLUIDOS_2025} de 2025)")
        for k, v in res.items():
            p(f"    {k:42s} MAPE {100 * np.mean(abs(R - v) / R):5.2f}%   "
              f"desvio médio {100 * np.mean((v - R) / R):+5.2f}%")
        cr = R.sum() / sum(serie[x - pd.DateOffset(years=1)] for x in idx) - 1
        p(f"    crescimento realizado {cr:.2%}; elasticidade implícita {cr / PIB_FOCUS[2025]:.2f}")
    else:
        p("\n[e] exercício de PIB x elasticidade não executado (RODAR_PIB_ELASTICIDADE = False).")

    with open("analises_complementares.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


# ==============================================================================
# 11) FIGURAS DO TCC
# ==============================================================================
def figuras_tcc(serie):
    import matplotlib.dates as md
    # Formato exigido pelo manual: sem grade, sem borda e sem título; eixos em linha preta de 1,5 pt;
    # fonte Arial (ou Liberation Sans, métrica equivalente) em tamanho que resulte em até 11 pt no documento.
    plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 18,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "black", "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5, "axes.grid": False})
    cor = {HW: "#ff7f0e", SARIMA: "#2ca02c", PROPHET: "#d62728", XGB: "#9467bd", NAIVE: "#1f77b4"}
    fbr = lambda v, d: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    fmi = FuncFormatter(lambda v, _: fbr(v / 1e6, 2))
    fmil = FuncFormatter(lambda v, _: fbr(v, 0))

    # Figura 1 — STL
    st = STL(serie, period=12, robust=True).fit()
    fig, ax = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True)
    for a, (y, lab, c, f) in zip(ax, [(serie, "Observado\n(milhões)", "black", fmi),
                                      (st.trend, "Tendência\n(milhões)", "#1f77b4", fmi),
                                      (st.seasonal, "Sazonal\n(veículos)", "#2ca02c", fmil),
                                      (st.resid, "Resíduo\n(veículos)", "#d62728", fmil)]):
        a.plot(y.index, y.values, color=c, lw=1.3); a.set_ylabel(lab)
        a.yaxis.set_major_formatter(f)
    ax[-1].set_xlabel("Ano"); fig.tight_layout(); fig.savefig("figura1_stl.png", dpi=300); plt.close(fig)

    # Figura 2 — previsões um passo à frente na janela de teste
    bt = pd.read_excel("serie_prevista.xlsx", sheet_name="Backtest_h1"); bt.index = pd.to_datetime(bt["Data"])
    fig, a = plt.subplots(figsize=(12, 6.2))
    a.plot(bt.index, bt["Real"], color="black", lw=2.4, marker="o", ms=4, label="Série real")
    for m, c in cor.items():
        a.plot(bt.index, bt[m], color=c, lw=1.4, marker=".", ms=5, label=m)
    a.yaxis.set_major_formatter(fmi); a.set_ylabel("Volume mensal (milhões de veículos)")
    a.xaxis.set_major_formatter(md.DateFormatter("%m/%Y"))
    a.xaxis.set_major_locator(md.MonthLocator(bymonth=[1, 4, 7, 10]))
    a.legend(frameon=False, fontsize=13, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig("figura2_janela_teste.png", dpi=300); plt.close(fig)

    # Figura 3 — MASE de cada modelo dividido pelo do naïve sazonal, por passo (horizonte de 12 meses)
    mp = pd.read_excel("serie_prevista.xlsx", sheet_name="MASE_por_passo").set_index("Passo")
    fig, a = plt.subplots(figsize=(12, 6.2))
    for m in [HW, SARIMA, PROPHET, XGB]:
        a.plot(mp.index, mp[m] / mp[NAIVE], color=cor[m], lw=2, marker="o", ms=6, label=m)
    a.axhline(1, color="black", ls="--", lw=1.2, label="Naïve sazonal (= 1)")
    a.set_xticks(range(1, 13)); a.set_xlabel("Passos à frente (meses)")
    a.set_ylabel("MASE do modelo ÷ MASE do naïve")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fbr(v, 1)))
    a.set_ylim(0, 1.1)
    a.legend(frameon=False, fontsize=13, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout(); fig.savefig("figura3_erro_por_passo.png", dpi=300); plt.close(fig)

    # Figura 4 — previsão de 12 meses com intervalo de 95% do Holt-Winters
    pf = pd.read_excel("serie_prevista.xlsx", sheet_name="Previsao_Futura"); pf.index = pd.to_datetime(pf["Data"])
    hw = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=12,
                              initialization_method="estimated").fit()
    fc = hw.forecast(12)
    sim = hw.simulate(12, repetitions=5000, error="add", anchor="end",
                      random_state=np.random.RandomState(SEED))
    lo, hi = sim.quantile(.025, axis=1), sim.quantile(.975, axis=1)
    fig, a = plt.subplots(figsize=(13, 5.6)); z = serie["2019-01":]
    a.plot(z.index, z.values, color="black", lw=1.5, label="Série real")
    a.axvspan(pd.Timestamp(COVID_INICIO), pd.Timestamp("2020-12-31"), color="gray", alpha=.15,
              label="Quebra pandêmica (mar–dez/2020)")
    a.fill_between(fc.index, lo, hi, color=cor[HW], alpha=.18, label="Intervalo de 95% (Holt-Winters)")
    for m, c in cor.items():
        a.plot(pf.index, pf[m], color=c, lw=1.6, marker=".", ms=5, label=m)
    a.axvline(serie.index[-1], color="black", ls=":", lw=1)
    a.yaxis.set_major_formatter(fmi); a.set_ylabel("Volume mensal (milhões de veículos)"); a.set_xlabel("Ano")
    a.legend(frameon=False, fontsize=11, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig("figura4_previsao.png", dpi=300); plt.close(fig)
    amp = 100 * (hi - lo) / fc
    print(f"Figuras do TCC gravadas. Amplitude do intervalo: {amp.iloc[0]:.1f}% (mês 1) a {amp.iloc[-1]:.1f}% (mês 12).")


# ==============================================================================
# 12) NÚMEROS CITADOS NO TEXTO DO TCC (conferência item a item)
# ==============================================================================
def numeros_do_texto(serie):
    """Imprime e grava em numeros_do_texto.txt todos os números do TCC que não
    aparecem diretamente nas tabelas, na ordem em que surgem no texto."""
    from scipy.stats import ttest_1samp
    from pmdarima.arima import nsdiffs
    out = []
    p = lambda s="": (print(s), out.append(s))
    tr = serie.iloc[:-N_TESTE]
    p("=" * 78); p(" NÚMEROS CITADOS NO TEXTO"); p("=" * 78)

    # Caracterização da série
    p(f"[1] Série: {serie.index[0]:%m/%Y} a {serie.index[-1]:%m/%Y}, {len(serie)} meses; "
      f"treino {len(tr)} meses, teste {N_TESTE} meses")
    p(f"    média mensal no triênio inicial: {serie.iloc[:36].mean():,.0f}; "
      f"no triênio final: {serie.iloc[-36:].mean():,.0f}")
    a19, a20 = serie[pd.Timestamp('2019-04-01')], serie[pd.Timestamp('2020-04-01')]
    p(f"    abril/2019 = {a19:,.0f}; abril/2020 = {a20:,.0f}; variação {100 * (a20 / a19 - 1):.1f}%")

    # Forças da STL (série original e série com mar-dez/2020 ajustados)
    ft, fs, _ = forca_stl(serie)
    st = STL(tr, period=12, robust=True).fit(); ajuste = st.trend + st.seasonal
    adj = serie.copy()
    meses = ajuste.index[(ajuste.index >= pd.Timestamp(COVID_INICIO)) & (ajuste.index <= pd.Timestamp('2020-12-01'))]
    adj[meses] = ajuste[meses]
    ft2, fs2, _ = forca_stl(adj)
    p(f"[2] Força da tendência / sazonalidade: série original {ft:.3f} / {fs:.3f}; "
      f"série ajustada {ft2:.3f} / {fs2:.3f}")

    # Testes de raiz unitária
    a = adfuller(serie, regression="ct", autolag="AIC")
    k = kpss(serie, regression="ct", nlags="auto")
    p(f"[3] ADF (constante e tendência; H0: raiz unitária): estatística {a[0]:.3f}, p = {a[1]:.3f}, "
      f"valor crítico 5% = {a[4]['5%']:.3f}")
    p(f"    KPSS (constante e tendência; H0: estacionária em torno de tendência): estatística {k[0]:.3f}, "
      f"p = {k[1]:.3f} (o statsmodels limita o p-valor a 0,10), valor crítico 5% = {k[3]['5%']:.3f}")
    for nome, x in (("série completa", serie), ("treino", tr)):
        p(f"    OCSB (H0: raiz unitária sazonal) e Canova-Hansen (H0: sazonalidade estável), {nome}: "
          f"D = {nsdiffs(x.values, m=12, test='ocsb')} e D = {nsdiffs(x.values, m=12, test='ch')}")

    # Resultados um passo à frente
    bt = pd.read_excel("serie_prevista.xlsx", sheet_name="Backtest_h1"); bt.index = pd.to_datetime(bt["Data"])
    mods = [HW, SARIMA, PROPHET, XGB]
    mae = lambda m: np.mean(np.abs(bt["Real"] - bt[m]))
    p(f"[4] Razão entre o MAE do Holt-Winters e o do naïve sazonal no teste: {mae(HW) / mae(NAIVE):.3f}")
    pe = bt[mods].sub(bt["Real"], axis=0).abs().div(bt["Real"], axis=0).mean(axis=1) * 100
    p(f"    erro percentual médio dos quatro modelos: jan/2024 {pe[pd.Timestamp('2024-01-01')]:.1f}%, "
      f"fev/2024 {pe[pd.Timestamp('2024-02-01')]:.1f}%")
    ea = bt[mods].sub(bt["Real"], axis=0).abs().sum(axis=1)
    p(f"    parcela do erro absoluto acumulado em dezembro a fevereiro: "
      f"{100 * ea[bt.index.month.isin([12, 1, 2])].sum() / ea.sum():.1f}%")
    e = (bt["Real"] - bt[HW]).values
    lb = acorr_ljungbox(e, lags=[6], return_df=True)["lb_pvalue"].iloc[0]
    t_v, p_v = ttest_1samp(e, 0.0)
    p(f"[5] Erros do Holt-Winters: Ljung-Box com 6 defasagens (H0: sem autocorrelação) p = {lb:.3f}; "
      f"erro médio {e.mean():,.0f} veículos, teste t (H0: média zero) p = {p_v:.3f}")

    # Horizonte de doze passos: erro relativo ao naïve por passo
    mp = pd.read_excel("serie_prevista.xlsx", sheet_name="MASE_por_passo").set_index("Passo")
    rel = mp[mods].div(mp[NAIVE], axis=0)
    p(f"[6] MASE relativo ao naïve, Holt-Winters: passo 1 = {rel[HW].iloc[0]:.2f}; "
      f"passo 12 = {rel[HW].iloc[-1]:.2f}; maior razão entre todos os modelos e passos = {rel.values.max():.2f}")

    with open("numeros_do_texto.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


# ==============================================================================
# 13) ESCOLHA DA CONFIGURAÇÃO DO PROPHET SÓ COM O TREINO (validação interna)
# ==============================================================================
N_VALID_INTERNA = 24   # últimos 24 meses do treino (dez/2020 a nov/2022)

def selecionar_config_prophet(serie):
    """Compara as três formas de tratar o calendário no Prophet usando apenas
    o treino: origem móvel de um passo sobre os últimos N_VALID_INTERNA meses
    do treino, com o MASE escalonado pelo naïve sazonal do treino interno.
    A janela de teste não é usada nesta escolha."""
    tr = serie.iloc[:-N_TESTE]
    ini = len(tr) - N_VALID_INTERNA
    esc = escala_mase(tr.iloc[:ini])
    res = {}
    for modo in ("nenhum", "nativo", "mensal"):
        prev = [m_prophet(tr.iloc[:t], 1, modo=modo)[0] for t in range(ini, len(tr))]
        res[modo] = metricas(tr.iloc[ini:].values, np.array(prev), esc)
    escolhido = min(res, key=lambda k: res[k]["MASE"])
    linhas = [f"Validação interna do Prophet ({tr.index[ini]:%m/%Y} a {tr.index[-1]:%m/%Y}, só treino)"]
    for k, v in res.items():
        linhas.append(f"    {k:7s} MAE {v['MAE']:,.0f}  RMSE {v['RMSE']:,.0f}  MAPE {v['MAPE']:.2f}%  MASE {v['MASE']:.3f}")
    linhas.append(f"    configuração escolhida: {escolhido}")
    print("\n".join(linhas))
    with open("selecao_prophet.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas))
    return escolhido


# ==============================================================================
# 14) MÉTODO DOS ESTUDOS (PIB x ELASTICIDADE) NA MESMA RÉGUA DOS MODELOS
# ==============================================================================
# Cada origem reproduz um estudo de estruturação: usa só o tráfego observado
# até a origem e o Boletim Focus daquela data, como se faz no valuation.
# Expectativas de crescimento do PIB (mediana Focus):
#   25/11/2022: 2,81% (2022) e 0,70% (2023) - valores informados na matéria da
#               B3 (Bora Investir, 05/12/2022) sobre o Focus da semana anterior;
#   24/11/2023: 2,84% (2023) e 1,50% (2024) - Focus R20231124.pdf (BCB).
FOCUS_ORIGENS = {"2022-11-01": {2022: 0.0281, 2023: 0.0070},
                 "2023-11-01": {2023: 0.0284, 2024: 0.0150}}
ELASTICIDADE_ESTUDOS = 1.0   # adotada nos estudos de tráfego da BR-381/262/MG/ES (Moraes, 2022)


def exercicio_metodo_estudos(serie):
    """Projeta, a partir de cada origem, os 12 meses seguintes pelo método dos
    estudos (mesmo mês do ano anterior x (1 + PIB esperado x elasticidade)) e
    compara com as previsões de 12 passos dos modelos feitas nas mesmas origens."""
    tr = serie.iloc[:-N_TESTE]
    esc = escala_mase(tr)
    ordem, _ = ordem_sarima(tr)
    modelos = construir_modelos(ordem)
    linhas = []
    for orig, pib in FOCUS_ORIGENS.items():
        t = serie.index.get_loc(pd.Timestamp(orig)) + 1
        trn, alvo = serie.iloc[:t], serie.iloc[t:t + 12]
        for nome, f in modelos.items():
            prev = np.asarray(f(trn, 12))
            linhas += [(orig, nome, d, r, prev[i]) for i, (d, r) in enumerate(alvo.items())]
        for d, r in alvo.items():
            base = serie[d - pd.DateOffset(years=1)]
            linhas.append((orig, "Método dos estudos (PIB x elasticidade)", d, r,
                           base * (1 + ELASTICIDADE_ESTUDOS * pib[d.year])))
    df = pd.DataFrame(linhas, columns=["origem", "modelo", "data", "real", "previsto"])
    nv = df[df.modelo == NAIVE]; mae_nv = (nv.real - nv.previsto).abs().mean()
    res = []
    for nome, g in df.groupby("modelo"):
        e = g.real - g.previsto
        res.append({"Método": nome, "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                    "MAPE (%)": 100 * (e.abs() / g.real).mean(), "MASE": e.abs().mean() / esc,
                    "MAE relativo ao naïve": e.abs().mean() / mae_nv,
                    "Desvio médio (%)": 100 * ((g.previsto - g.real) / g.real).mean()})
    tab = pd.DataFrame(res).sort_values("MAE").set_index("Método")
    txt = ("Método dos estudos x modelos — origens nov/2022 e nov/2023, 12 meses à frente "
           "(24 previsões por método)\n" + tab.round(3).to_string())
    print(txt)
    with open("exercicio_metodo_estudos.txt", "w", encoding="utf-8") as fh:
        fh.write(txt)
    df.to_csv("exercicio_metodo_estudos.csv", index=False)
    return tab


if __name__ == "__main__":
    PROPHET_FERIADOS = selecionar_config_prophet(_serie_principal())
    main()                       # rodada principal (itens 1 a 7)
    serie = _serie_principal()
    analises_complementares(serie)
    figuras_tcc(serie)
    numeros_do_texto(serie)
    exercicio_metodo_estudos(serie)
