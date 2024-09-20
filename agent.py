from datetime import date, datetime
from typing import Annotated, Optional
from typing_extensions import TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import AnyMessage, add_messages

load_dotenv()


def get_data(data_path):
    df = pd.read_csv(data_path)
    df["data_index"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("data_index")
    df = df.sort_index()
    return df


def get_by_states(df, states):
    return df[((df["client"] == states[0]) & (df["server"] == states[1]))]


# DADOS do trabalhdo

df = get_data("data/dados.csv")


# DEFINIÇÃO DAS FUNÇÕES DE APOIO


@tool
def calcula_qoes_cliente(
    cliente: str,
    data_inicio: Annotated[
        Optional[date | datetime],
        "Data de inicio que se deseja analisar os dados. Os dados são filtrados para serem analisados a partir desta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[0]),
    data_fim: Annotated[
        Optional[date | datetime],
        "Data de fim que se deseja analisar os dados. Os dados são filtrados para serem analisados até esta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[-1]),
):
    """Função calcula o QoE entre um cliente especifico e todos os servidores que podem se conectar a ele. A função retorna o QoE de todos os pares possíveis com o cliente especificado.
    Utilize esta função para obter os diferentes valores de QoE para cada servidor conectado a um cliente específico.
    """

    df_filter = df["client"] == cliente
    data = df[df_filter]

    if len(data) == 0:
        return f"O cliente {cliente} não existe na rede."

    data = data[(data.index >= data_inicio) & (data.index <= data_fim)]
    if len(data) == 0:
        return f"Não há dados para essas datas."

    data = data.groupby(["client", "server"]).agg({"bitrate": "mean", "rtt": "mean"})
    data.loc[:, "qoe"] = data["bitrate"] / data["rtt"]
    data = data[["qoe"]]
    return data.to_dict()


@tool
def calcula_qoes_servidor(
    servidor: str,
    data_inicio: Annotated[
        Optional[date | datetime],
        "Data de inicio que se deseja analisar os dados. Os dados são filtrados para serem analisados a partir desta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[0]),
    data_fim: Annotated[
        Optional[date | datetime],
        "Data de fim que se deseja analisar os dados. Os dados são filtrados para serem analisados até esta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[-1]),
):
    """Função calcula o QoE entre um servidor especifico e todos os clientes que podem se conectar a ele. A função retorna o QoE de todos os pares possíveis com o servidor especificado.
    Utilize esta função para obter os diferentes valores de QoE para cada cliente conectado a um servidor específico.
    """

    df_filter = df["server"] == servidor
    data = df[df_filter]

    if len(data) == 0:
        return f"O servidor {servidor} não existe na rede."

    data = data[(data.index >= data_inicio) & (data.index <= data_fim)]
    if len(data) == 0:
        return f"Não há dados para essas datas."

    data = data.groupby(["client", "server"]).agg({"bitrate": "mean", "rtt": "mean"})
    data.loc[:, "qoe"] = data["bitrate"] / data["rtt"]
    data = data[["qoe"]]
    return data.to_dict()


@tool
def get_bitrate_latencia(
    cliente: str,
    servidor: str,
    data_inicio: Annotated[
        Optional[date | datetime],
        "Data de inicio que se deseja analisar os dados. Os dados são filtrados para serem analisados a partir desta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[0]),
    data_fim: Annotated[
        Optional[date | datetime],
        "Data de fim que se deseja analisar os dados. Os dados são filtrados para serem analisados até esta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[-1]),
):
    """Função retorna o bitrate e a latencia media ao longo do tempo (data_inicio até data_fim) para um par cliente servidor. Use esta função para obter os valores de bitrate e latencia para o calculo do QoE.
    Use está função somente quando precisar calcular o QoE de um único par cliente servidor.
    """

    df_filter = (df["client"] == cliente) & (df["server"] == servidor)
    data = df[df_filter]

    if len(data) == 0:
        return f"O cliente {cliente} não se conecta ao servidor {servidor}"

    data = data[(data.index >= data_inicio) & (data.index <= data_fim)]
    if len(data) == 0:
        return f"Não há dados para essas datas."

    return (
        data.groupby(["client", "server"])
        .agg({"bitrate": "mean", "rtt": "mean"})
        .rename(columns={"rtt": "latencia"})
        .to_dict()
    )


@tool
def get_bitrates_latencias_cliente(
    cliente: str,
    data_inicio: Annotated[
        Optional[date | datetime],
        "Data de inicio que se deseja analisar os dados. Os dados são filtrados para serem analisados a partir desta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[0]),
    data_fim: Annotated[
        Optional[date | datetime],
        "Data de fim que se deseja analisar os dados. Os dados são filtrados para serem analisados até esta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[-1]),
):
    """Função retorna o bitrate e a latencia media ao longo do tempo (data_inicio até data_fim) do cliente especificado em relação a todos os servidores que podem se conectar a ele.
    Use esta função para obter os valores de bitrate e latencia de um cliente específico em relação a todos os seus servidores para o calculo dos QoEs. Use está função quando for necessário calcular o QoE em relação a todos os servidores do cliente.
    """

    df_filter = df["client"] == cliente
    data = df[df_filter]

    if len(data) == 0:
        return f"O cliente {cliente} não existe na rede."

    data = data[(data.index >= data_inicio) & (data.index <= data_fim)]
    if len(data) == 0:
        return f"Não há dados para essas datas."

    return (
        data.groupby(["client", "server"])
        .agg({"bitrate": "mean", "rtt": "mean"})
        .rename(columns={"rtt": "latencia"})
        .to_dict()
    )


@tool
def get_bitrates_latencias_servidor(
    servidor: str,
    data_inicio: Annotated[
        Optional[date | datetime],
        "Data de inicio que se deseja analisar os dados. Os dados são filtrados para serem analisados a partir desta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[0]),
    data_fim: Annotated[
        Optional[date | datetime],
        "Data de fim que se deseja analisar os dados. Os dados são filtrados para serem analisados até esta data. Utilize o formato YYYY-MM-DD HH:MM:SS",
    ] = str(df.index[-1]),
):
    """Função retorna o bitrate e a latencia media ao longo do tempo (data_inicio até data_fim) do servidor especificado em relação a todos os clientes que podem se conectar a ele.
    Use esta função para obter os valores de bitrate e latencia de um servidor específico em relação a todos os seus clientes para o calculo dos QoEs. Use está função quando for necessário calcular o QoE em relação a todos os clientes do servidor.
    """

    df_filter = df["server"] == servidor
    data = df[df_filter]

    if len(data) == 0:
        return f"O servidor {servidor} não existe na rede."

    data = data[(data.index >= data_inicio) & (data.index <= data_fim)]
    if len(data) == 0:
        return f"Não há dados para essas datas."

    return (
        data.groupby(["client", "server"])
        .agg({"bitrate": "mean", "rtt": "mean"})
        .rename(columns={"rtt": "latencia"})
        .to_dict()
    )


@tool
def calcula_qoe(bitrate: float, latencia: float):
    """Função que calcula o QoE dado o bitrate e a latencia da conexão. O QoE é calculado como a razão entre o bitrate e a latência de uma conexão."""
    qoe = bitrate / latencia
    return f"O QoE calculado é {qoe}"


@tool
def media_qoe(lista_valores_qoe: list):
    """Calcula a media dos valores de qoe na lista fornecida"""
    media = np.array(lista_valores_qoe).mean()
    return f"A média dos valores dos QoEs é {media}"


@tool
def variancia_qoe(lista_valores_qoe: list):
    """Calcula a variancia dos valores de qoe na lista fornecida. Use para analisar a consistencia da rede"""
    var = np.var(np.array(lista_valores_qoe))
    return f"A variancia dos valores dos QoEs é {var}"


def get_agent():

    # DADOS DE CONTEXTO PARA O PROMPT
    clientes = df["client"].unique().tolist()
    servidores = df["server"].unique().tolist()
    cliente_servidor = list(df.groupby(["client", "server"]).groups.keys())

    model_prompt = (
        "Você é um assistente de suporte de uma empresa que mantém uma rede de transmissão de vídeo. "
        "Você deve responder perguntas sobre a qualidade de experiência (QoE) da rede de transmissão dos vídeos. "
        "A qualidade de transmissão é analisada através de dados do bitrate e latência entre os clientes e servidores da rede. "
        "A qualidade de experiência (QoE) é determinada pela razão entre o bitrate e a latência de um par cliente servidor. "
        "O QoE de um cliente especifico é dado pela média dos QoEs deste cliente em relação a cada servidor que ele se conecta. "
        "O QoE de um servidor especifico é dado pela média dos QoEs deste servicor em relação a cada cliente conectado a ele. "
        "Cada cliente e servidor é representado pela sigla do estado onde estão localizados. "
        f"Você tem acesso a dados do dia {str(df.index[0])} ao dia {str(df.index[-1])}. "
        f"A rede é composta pelos seguintes clientes: {clientes}. E pelos seguintes servidores: {servidores}. Os seguintes pares de cliente servidor estão presentes na rede: {cliente_servidor}. "
        "Use as ferramentas fornecidas para buscar informações sobre os clientes e servidores da rede de transmissão e, assim, responder às consultas dos usuários. "
        "Para cada ferramenta escolhida justifique sua escolha descrevendo o que você deseja obter como retorno da ferramenta. Não cite a ferramenta explicitamente, somente o que você deseja obter e o motivo. "
        "Caso a pergunta não seja sobre a rede de transmissão diga que não atua fora do escopo deste assunto. "
        "Caso sejam necessárias mais informações para responder a pergunta, você pode pedir para o usuário entrar com as informações necessárias. "
        "Caso não consiga responder à consulta com as ferramentas disponibilizadas, diga apenas que é incapaz de responder. Não tente inventar uma resposta sem dados para justificá-la. "
        "Gere o texto da resposta sem utilizar marcações LaTeX, markdown, html etc. "
        "É essencial e imprescindível que você explique seu raciocínio e todos os passos para chegar à resposta final. "
    )

    # FUNÇÕES UTILIZADAS
    tools = [
        get_bitrate_latencia,
        calcula_qoe,
        media_qoe,
        variancia_qoe,
        get_bitrates_latencias_cliente,
        get_bitrates_latencias_servidor,
    ]  # , calcula_qoes_cliente, calcula_qoes_servidor]

    # ESTADO
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # AGENTE
    def agent(state: AgentState, config: RunnableConfig):

        while True:
            system_prompt = SystemMessage(model_prompt)
            response = model.invoke([system_prompt] + state["messages"], config)

            if (len(response.tool_calls) == 0 or not response.tool_calls) and len(
                response.content
            ) == 0:
                messages = state["messages"] + [
                    ("user", "Gere uma resposta real e válida.")
                ]
                state = {"messages": messages}
            else:
                break

        return {"messages": [response]}

    # LLM com Tool Calling
    model = ChatOpenAI(model="gpt-4o-mini")
    model = model.bind_tools(tools)

    # Nó que executa as funções
    tool_node = ToolNode(tools=tools)

    # GRAFO
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    # Módulo para salvar o histórico
    memory = MemorySaver()
    agent_graph = builder.compile(checkpointer=memory)

    return agent_graph


agent_graph = get_agent()
