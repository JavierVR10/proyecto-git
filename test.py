import streamlit as st

# Debe ser el primer comando de Streamlit en el script
st.set_page_config(page_title="Asistente de Base de Datos de Procesadores", layout="wide")

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

import pymysql
from sqlalchemy import create_engine, text

# Configuración del modelo y la base de datos
llm = ChatOllama(model='llama3.2:3b', base_url='http://localhost:11434')

db_uri = 'mysql+pymysql://root:javi3rJA@localhost:3306/procesadores'
db = SQLDatabase.from_uri(db_uri)

# Función para obtener datos de la capa semántica
def get_semantic_info():
    """Consulta la capa semántica y devuelve datos de ejemplo."""
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='javi3rJA',
            database='procesadores',
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM procesadores LIMIT 5;")
            result = cursor.fetchall()
        connection.close()
        return result
    except Exception as e:
        return f"Error al conectar con la base de datos: {e}"

# Instrucciones para el agente
SQL_PREFIX = """You are an assistant that helps users understand and query databases. 
You can provide explanations about the database structure, help with SQL queries, and explain concepts in natural language.

If the user asks how to structure a query in BigQuery or any other SQL database, you should provide a clear and concise example, explaining each part of the query.

For example, if the user asks "How do I structure a query in BigQuery?", you should respond with something like:

"In BigQuery, a basic SQL query follows this structure:
sql
SELECT *
FROM `CELULARES.GAMA-ALTA.PROCESADOR_2022_Qualcomm`  
; or
sql
SELECT *
FROM CELULARES.GAMA-ALTA.PROCESADOR_2022_Qualcomm; 

The MySQL database "procesadores" contains an internal semantic layer stored in a table also called "procesadores".
This table has the following columns: Proyecto, Esquema, Tabla, Campo, TipoDato, and Concepto.
This internal layer is used solely to help you understand the user's question and to generate a natural language explanation.

Similarly, if a user asks "What does the field 'Modelo' in the table PROCESADOR_2022_Qualcomm mean?",
you must provide an explanation of the field "Modelo" using the semantic layer's "Concepto" data, without showing any SQL query.

Your answer must be clear and solely in natural language, without revealing any details of the internal semantic layer.
Your answer must be clear and only in natural language, always relying on the semantic layer in the concept part, if you can rely on the rest it is fine, but the main thing is the Concept Column.

If they ask about where to request permissions to see the team's dashboard, you will provide them with this link https://www.youtube.com/watch?v=6DTWH9kYAiY
you will explain that here with VPN will have to raise a ticket where they will have to put this AD Group AD_GROUP_PROCESS_BOT

And with this you will be able to have access.
"""

# Crear el agente SQL con herramientas
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
system_message = SystemMessage(content=SQL_PREFIX)
agent_executor = create_react_agent(llm, tools, state_modifier=system_message, debug=False)

def handle_question(question):
    """Procesa la pregunta del usuario y devuelve la respuesta."""
    try:
        response = agent_executor.invoke({"messages": [HumanMessage(content=question)]})
        if isinstance(response, dict) and "messages" in response:
            for msg in response["messages"]:
                if isinstance(msg, AIMessage) and msg.content.strip():
                    return msg.content.strip()
        return "No se pudo interpretar la respuesta. Verifica tu consulta."
    except Exception as e:
        return f"Ocurrió un error al procesar tu consulta: {str(e)}"

# Inicializar historial en session_state
if "history" not in st.session_state:
    st.session_state.history = []

# Interfaz en Streamlit
st.title("🤖 Asistente de Base de Datos de Procesadores")
st.write("¡Hola! Soy un asistente de IA que te ayuda a entender y consultar la base de datos de procesadores. Hazme una pregunta.")

# Entrada de texto para la pregunta del usuario
user_question = st.text_input("Escribe tu pregunta aquí:")

if user_question:
    with st.spinner("Procesando tu pregunta..."):
        semantic_info = get_semantic_info()
        if isinstance(semantic_info, list):
            response = handle_question(user_question)
        else:
            response = semantic_info  # Si hay error en la BD, lo mostramos directamente

        # Guardar en historial
        st.session_state.history.append({"question": user_question, "answer": response})

    # Mostrar respuesta
    st.write("### Respuesta del Bot 🤖:")
    st.write(response)

# Mostrar historial de preguntas y respuestas
st.write("## Historial de Preguntas 📜")
for entry in st.session_state.history[::-1]:  # Mostramos de la más reciente a la más antigua
    with st.expander(f"Pregunta: {entry['question']}"):
        st.write(f"**Respuesta:** {entry['answer']}")
