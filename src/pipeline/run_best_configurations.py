from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from src.config.parser import config_treatment
from src.pipeline.runner import build_input_list
from src.scoring.objective import simulation_function


COLUNAS_METADATA = {
    "pasta_saida",
    "etapa",
    "arquivo_origem",
    "simulacao_escolhida",
    "simulacao_escolhida_treinamento",
    "simulacao_escolhida_final",
    "bnx_mais_importante",
    "bnx_train",
    "bnx_test",
    "euclidiana",
    "euclidiana_treinamento",
    "euclidiana_teste",
}


def _ler_csv_robusto(caminho: str | Path) -> pd.DataFrame:
    caminho = Path(caminho)

    try:
        return pd.read_csv(caminho)
    except Exception:
        return pd.read_csv(caminho, sep=";")


def _valor_texto(valor: Any, default: str = "") -> str:
    if pd.isna(valor):
        return default
    return str(valor).strip()


def _extrair_parametros_linha(
    linha: Mapping[str, Any],
    colunas_ignoradas: Iterable[str] = COLUNAS_METADATA,
) -> Dict[str, float]:
    """
    Extrai somente as colunas que representam parâmetros otimizados.

    Exemplo esperado:
    EMFL, FLSH, FLSD, SDPM, FLLF, LFMAX, SLAVR, ...
    """
    ignoradas = set(colunas_ignoradas)
    parametros: Dict[str, float] = {}

    for coluna, valor in linha.items():
        if coluna in ignoradas:
            continue

        if pd.isna(valor):
            continue

        try:
            parametros[str(coluna)] = float(valor)
        except Exception:
            # Caso apareça alguma coluna textual não prevista,
            # ela será ignorada em vez de quebrar a execução.
            continue

    if not parametros:
        raise ValueError(
            "Nenhum parâmetro numérico foi encontrado na linha do resumo. "
            f"Colunas disponíveis: {list(linha.keys())}"
        )

    return parametros


def _obter_bnx_mais_importante(linha: Mapping[str, Any]) -> str:
    """
    Usa preferencialmente a coluna bnx_mais_importante.

    Regras de segurança:
    - treinamento: fallback para bnx_train
    - pos_treinamento: fallback para bnx_test
    """
    bnx = _valor_texto(linha.get("bnx_mais_importante"))

    if bnx:
        return bnx

    etapa = _valor_texto(linha.get("etapa")).lower()

    if etapa == "treinamento":
        bnx = _valor_texto(linha.get("bnx_train"))
    elif etapa in {"pos_treinamento", "pós_treinamento", "pos-treinamento"}:
        bnx = _valor_texto(linha.get("bnx_test"))

    if not bnx:
        raise ValueError(
            "Não foi possível identificar o BNX da linha. "
            "Esperado 'bnx_mais_importante', ou fallback para 'bnx_train'/'bnx_test'."
        )

    return bnx


def _obter_simulacao_escolhida(linha: Mapping[str, Any]) -> str:
    simulacao = _valor_texto(linha.get("simulacao_escolhida"))

    if simulacao:
        return simulacao

    simulacao = _valor_texto(linha.get("simulacao_escolhida_treinamento"))
    if simulacao:
        return simulacao

    simulacao = _valor_texto(linha.get("simulacao_escolhida_final"))
    if simulacao:
        return simulacao

    return ""


def run_best_configurations_from_summary(
    *,
    arq_config: str | Path = "configs/StartValues_bean.config",
    arq_resumo: str | Path = "resumo_melhores_configuracoes.csv",
    arq_saida: str | Path = "evaluate_melhores_configuracoes.csv",
) -> pd.DataFrame:
    """
    Executa uma simulação DSSAT para cada linha do arquivo resumo_melhores_configuracoes.csv.

    Para cada linha:
    - pega os parâmetros otimizados;
    - pega o BNX em bnx_mais_importante;
    - roda DSSAT usando a pipeline existente;
    - coleta o Evaluate.OUT;
    - adiciona metadados da linha original;
    - salva tudo em CSV.
    """
    arq_config = Path(arq_config)
    arq_resumo = Path(arq_resumo)
    arq_saida = Path(arq_saida)

    if not arq_config.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {arq_config}")

    if not arq_resumo.exists():
        raise FileNotFoundError(f"Arquivo de resumo não encontrado: {arq_resumo}")

    cfg = config_treatment(arq_config)
    input_base = build_input_list(cfg)

    resumo = _ler_csv_robusto(arq_resumo)

    if resumo.empty:
        raise ValueError(f"O arquivo de resumo está vazio: {arq_resumo}")

    resultados: list[pd.DataFrame] = []

    for idx, row in resumo.iterrows():
        linha = row.to_dict()

        etapa = _valor_texto(linha.get("etapa"), default="sem_etapa")
        bnx_mais_importante = _obter_bnx_mais_importante(linha)
        simulacao_escolhida = _obter_simulacao_escolhida(linha)

        parametros = _extrair_parametros_linha(linha)

        bnx_label = Path(bnx_mais_importante).stem

        input_list = dict(input_base)
        input_list["bnxFile"] = bnx_mais_importante
        input_list["bnxLabel"] = bnx_label

        # Mantém bnxTest coerente quando a linha for de pós-treinamento.
        # Isso não atrapalha o treinamento e ajuda caso algum trecho da pipeline consulte bnxTest.
        if "bnx_test" in linha and not pd.isna(linha["bnx_test"]):
            input_list["bnxTest"] = str(linha["bnx_test"])

        template_id = f"bestcfg_{idx + 1:04d}_{etapa}_{bnx_label}"

        print(
            f"[{idx + 1}/{len(resumo)}] Rodando DSSAT | "
            f"etapa={etapa} | bnx={bnx_mais_importante} | "
            f"simulacao={simulacao_escolhida}"
        )

        evaluate = simulation_function(
            param_sim=parametros,
            template_id=template_id,
            input_list=input_list,
        )

        if evaluate is None or evaluate.empty:
            print(f"  Aviso: Evaluate vazio para linha {idx + 1}.")
            continue

        evaluate = evaluate.copy()

        # Metadados principais solicitados
        evaluate["linha_resumo"] = idx + 1
        evaluate["etapa"] = etapa
        evaluate["bnx_mais_importante"] = bnx_mais_importante
        evaluate["simulacao_escolhida"] = simulacao_escolhida

        # Metadados auxiliares, caso existam no resumo
        for col in [
            "pasta_saida",
            "arquivo_origem",
            "bnx_train",
            "bnx_test",
            "euclidiana",
            "euclidiana_treinamento",
            "euclidiana_teste",
        ]:
            if col in linha:
                evaluate[col] = linha[col]

        # Também salva os parâmetros usados em cada simulação
        for nome_parametro, valor_parametro in parametros.items():
            evaluate[f"param_{nome_parametro}"] = valor_parametro

        resultados.append(evaluate)

    if not resultados:
        raise RuntimeError(
            "Nenhuma simulação gerou Evaluate válido. "
            "Verifique os BNX, o executável DSSAT e os caminhos do arquivo .config."
        )

    saida = pd.concat(resultados, ignore_index=True)

    arq_saida.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(arq_saida, index=False, encoding="utf-8-sig")

    print(f"\nArquivo gerado com sucesso: {arq_saida}")
    print(f"Total de linhas salvas: {len(saida)}")

    return saida