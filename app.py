import streamlit as st
import ollama

st.set_page_config(page_title="Simplificador Financeiro", page_icon="📈")

st.title("Simplificador Financeiro 📈")
st.write("Converse com a IA para simplificar 'economês' e resolver cálculos matemáticos passo a passo.")

# 1. Inicializa a memória da conversa no Streamlit
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# 2. Exibe o histórico inteiro da conversa na tela
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Barra de digitação do chat no estilo ChatGPT
if prompt_usuario := st.chat_input("Ex: O que é CDI?"):
    
    # Adiciona a pergunta do usuário na tela e na memória
    st.chat_message("user").markdown(prompt_usuario)
    st.session_state.mensagens.append({"role": "user", "content": prompt_usuario})

    # 4. Gera a resposta da IA e adiciona na tela e na memória
    with st.chat_message("assistant"):
        with st.spinner("Processando..."):
            try:
                # O segredo está aqui: enviamos a lista inteira (st.session_state.mensagens)
                # Assim a IA lê tudo o que foi dito antes e mantém o contexto.
                resposta = ollama.chat(
                    model='educador-financeiro', 
                    messages=st.session_state.mensagens
                )
                
                texto_resposta = resposta['message']['content']
                st.markdown(texto_resposta)
                
                # Salva a resposta da IA na memória para o próximo turno
                st.session_state.mensagens.append({"role": "assistant", "content": texto_resposta})
                
            except Exception as e:
                st.error(f"Erro: {e}. Verifique se o Ollama está rodando.")