#!/usr/bin/env python3
"""
Agente de IA para Análise de Notas Fiscais CSV
Usando LangChain + Pandas com modelo local Llama via Ollama API (otimizado)
"""

import os
import zipfile
import pandas as pd
import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
from langchain.agents.agent_types import AgentType
import tempfile
from pathlib import Path

class NotasFiscaisAnalyzer:
    def __init__(self):
        """
        Inicializa o analisador de notas fiscais com Llama via Ollama
        """
        self.df_cabecalho = None
        self.df_itens = None
        self.agent = None

        self.llm = Ollama(
            model="llama3",
            base_url="http://127.0.0.1:11434"
        )

    def carregar_dados(self, csv_cabecalho_path, csv_itens_path):
        self.df_cabecalho = pd.read_csv(csv_cabecalho_path)
        self.df_itens = pd.read_csv(csv_itens_path)
        st.success("Arquivos carregados com sucesso!")

    def criar_agente(self, usar_itens=False):
        df = self.df_itens if usar_itens else self.df_cabecalho

        prefixo = (
            "Você é um analista de notas fiscais brasileiras."
            " Quando receber uma pergunta, responda com base nos dados do DataFrame que lhe foi fornecido."
            " Use as funções do pandas como groupby(), sum(), mean() e outras conforme necessário."
            " Sempre retorne os resultados com valores monetários no formato R$ X.XXX,XX e datas no formato DD/MM/AAAA."
            " Caso encontre um erro, retorne um aviso e tente outra abordagem."
        )

        self.agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=[df],
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            prefix=prefixo,
            allow_dangerous_code=True
        )

    def executar_pergunta(self, pergunta):
        if not self.agent:
            st.error("Agente não foi criado ainda.")
            return "Agente não disponível."

        try:
            resposta = self.agent.run(pergunta)
            return resposta
        except Exception as e:
            return f"Erro ao executar a pergunta: {str(e)}"


def main():
    st.set_page_config(page_title="Analisador de Notas Fiscais", page_icon="📊")
    st.title("Analisador de Notas Fiscais com IA (LLM Local)")
    st.markdown("Faça upload dos arquivos CSV de cabeçalho e itens")

    csv_cabecalho = st.file_uploader("Arquivo CSV - Cabeçalho", type="csv")
    csv_itens = st.file_uploader("Arquivo CSV - Itens", type="csv")

    pergunta = st.text_input("Digite sua pergunta sobre os dados:")
    usar_itens = st.checkbox("Usar dados de itens?", value=False)

    if csv_cabecalho and csv_itens:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_cab:
            tmp_cab.write(csv_cabecalho.read())
            path_cab = tmp_cab.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_it:
            tmp_it.write(csv_itens.read())
            path_it = tmp_it.name

        analisador = NotasFiscaisAnalyzer()
        analisador.carregar_dados(path_cab, path_it)
        analisador.criar_agente(usar_itens=usar_itens)

        if pergunta:
            resposta = analisador.executar_pergunta(pergunta)
            st.subheader("Resposta da IA:")
            st.write(resposta)

if __name__ == "__main__":
    main()
