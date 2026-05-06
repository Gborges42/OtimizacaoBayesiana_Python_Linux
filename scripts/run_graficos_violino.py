import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. Ler o arquivo
# =========================
arquivo_csv = "./analysis/evaluate_por_tratamento.csv"
df = pd.read_csv(arquivo_csv)

# =========================
# 2. Definir os pares
# =========================
# S = simulação
# M = medição
pares = {
    "MDAP": ("MDAPS", "MDAPM"),
    "ADAP": ("ADAPS", "ADAPM"),
    "PD1P": ("PD1PS", "PD1PM"),
    "PDFP": ("PDFPS", "PDFPM"),
    "HWAM": ("HWAMS", "HWAMM"),
    "PWAM": ("PWAMS", "PWAMM"),
    "HWUM": ("HWUMS", "HWUMM"),
    "H#AM": ("H#AMS", "H#AMM"),
    "H#UM": ("H#UMS", "H#UMM"),
    "THAM": ("THAMS", "THAMM"),
    "LAIX": ("LAIXS", "LAIXM"),
    "HIAM": ("HIAMS", "HIAMM"),
}

# =========================
# 3. Calcular o MAPE (%)
# =========================
resultado = df[["etapa", "RUN"]].copy()

for variavel, (col_s, col_m) in pares.items():
    medicao = df[col_m].replace(0, np.nan)  # evita divisão por zero
    resultado[f"MAPE_{variavel}"] = (
        np.abs((df[col_s] - df[col_m]) / medicao) * 100
    )

# =========================
# 4. Transformar em formato longo
# =========================
df_long = resultado.melt(
    id_vars=["etapa", "RUN"],
    value_vars=[f"MAPE_{v}" for v in pares.keys()],
    var_name="variavel",
    value_name="MAPE"
)

df_long["variavel"] = df_long["variavel"].str.replace("MAPE_", "", regex=False)
df_long = df_long.dropna(subset=["MAPE"]).copy()

# =========================
# 5. Definir ordem
# =========================
ordem_variaveis = ["MDAP", "ADAP", "PD1P", "PDFP"]
ordem_simulacoes = df["etapa"].drop_duplicates().tolist()

# =========================
# 6. Montar dados para o gráfico
# =========================
dados_plot = []
labels_plot = []
posicoes = []

pos = 1
espaco_entre_grupos = 1  # espaço visual entre variáveis
centros_grupos = []

for variavel in ordem_variaveis:
    inicio_grupo = pos
    
    for sim in ordem_simulacoes:
        valores = df_long.loc[
            (df_long["variavel"] == variavel) &
            (df_long["etapa"] == sim),
            "MAPE"
        ].values
        
        dados_plot.append(valores)
        labels_plot.append(f"{sim}")
        posicoes.append(pos)
        pos += 1
    
    fim_grupo = pos - 1
    centros_grupos.append((inicio_grupo + fim_grupo) / 2)
    pos += espaco_entre_grupos

# =========================
# 7. Criar gráfico
# =========================
plt.figure(figsize=(16, 7))

plt.violinplot(
    dados_plot,
    positions=posicoes,
    showmeans=True,
    showmedians=True
)

# rótulos das simulações
plt.xticks(posicoes, labels_plot, rotation=45, ha="right")

# nomes das variáveis acima dos grupos
ymax = max([np.nanmax(vals) for vals in dados_plot if len(vals) > 0])
for centro, variavel in zip(centros_grupos, ordem_variaveis):
    plt.text(
        centro,
        ymax * 1.05,
        variavel,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

# linhas separando os grupos
for i in range(len(ordem_variaveis) - 1):
    ultimo_do_grupo = posicoes[(i + 1) * len(ordem_simulacoes) - 1]
    primeiro_prox = posicoes[(i + 1) * len(ordem_simulacoes)]
    x_sep = (ultimo_do_grupo + primeiro_prox) / 2
    plt.axvline(x=x_sep, linestyle="--", alpha=0.5)

plt.ylabel("MAPE (%)")
plt.xlabel("Simulação dentro de cada variável")
plt.title("Distribuição do erro MAPE por variável e simulação")
plt.tight_layout()

# =========================
# 8. Salvar e mostrar
# =========================
nome_saida = "./graficos/violino_mape_variavel.png"
plt.savefig(nome_saida, dpi=400, bbox_inches="tight")
plt.show()
plt.close()

print(f"Gráfico salvo como: {nome_saida}")