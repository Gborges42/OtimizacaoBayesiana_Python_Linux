from pathlib import Path
import pandas as pd


PASTA_OUTPUT = Path("../output")
ARQUIVO_SAIDA = "./analysis/melhores_por_tratamento.csv"


def ler_csv(caminho: Path) -> pd.DataFrame:
    """
    Lê CSV tentando separadores comuns.
    """
    try:
        return pd.read_csv(caminho)
    except Exception:
        return pd.read_csv(caminho, sep=";")


def processar_arquivo(caminho_csv: Path, pasta_simulacao: str, etapa: str) -> dict:
    df = ler_csv(caminho_csv)

    if df.empty:
        raise ValueError(f"Arquivo vazio: {caminho_csv}")

    # Normalmente existe apenas uma linha com a melhor configuração
    linha = df.iloc[0].to_dict()

    resultado = {
        "pasta_saida": pasta_simulacao,
        "etapa": etapa,
        "arquivo_origem": caminho_csv.name,
    }

    if etapa == "treinamento":
        resultado["simulacao_escolhida"] = linha.get("simulacao_escolhida_treinamento")
        resultado["bnx_mais_importante"] = linha.get("bnx_train")
        resultado["euclidiana"] = linha.get("euclidiana_treinamento")

    elif etapa == "pos_treinamento":
        resultado["simulacao_escolhida"] = linha.get("simulacao_escolhida_final")

        # Atenção:
        # No pós-treinamento, o BNX mais importante é o bnx_test
        resultado["bnx_mais_importante"] = linha.get("bnx_test")

        resultado["bnx_train"] = linha.get("bnx_train")
        resultado["euclidiana_treinamento"] = linha.get("euclidiana_treinamento")
        resultado["euclidiana_teste"] = linha.get("euclidiana_teste")

    # Colunas que não são variáveis otimizadas
    colunas_ignoradas = {
        "simulacao_escolhida_treinamento",
        "simulacao_escolhida_final",
        "bnx_train",
        "bnx_test",
        "euclidiana_treinamento",
        "euclidiana_teste",
    }

    # As demais colunas são tratadas como variáveis otimizadas
    for coluna, valor in linha.items():
        if coluna not in colunas_ignoradas:
            resultado[coluna] = valor

    return resultado


def main():
    if not PASTA_OUTPUT.exists():
        raise FileNotFoundError(f"A pasta '{PASTA_OUTPUT}' não foi encontrada.")

    resultados = []

    for pasta in sorted(PASTA_OUTPUT.iterdir()):
        if not pasta.is_dir():
            continue

        caminho_treinamento = pasta / "melhor_configuracao_treinamento.csv"
        caminho_pos_treinamento = pasta / "melhor_configuracao_pos_treinamento.csv"

        if caminho_treinamento.exists():
            try:
                resultados.append(
                    processar_arquivo(
                        caminho_csv=caminho_treinamento,
                        pasta_simulacao=pasta.name,
                        etapa="treinamento",
                    )
                )
            except Exception as erro:
                print(f"Erro ao processar {caminho_treinamento}: {erro}")

        if caminho_pos_treinamento.exists():
            try:
                resultados.append(
                    processar_arquivo(
                        caminho_csv=caminho_pos_treinamento,
                        pasta_simulacao=pasta.name,
                        etapa="pos_treinamento",
                    )
                )
            except Exception as erro:
                print(f"Erro ao processar {caminho_pos_treinamento}: {erro}")

    if not resultados:
        print("Nenhum arquivo de configuração foi encontrado.")
        return

    df_saida = pd.DataFrame(resultados)

    df_saida.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")

    print(f"Arquivo gerado com sucesso: {ARQUIVO_SAIDA}")
    print(f"Total de registros salvos: {len(df_saida)}")


if __name__ == "__main__":
    main()