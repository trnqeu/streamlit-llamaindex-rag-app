import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import CitationQueryEngine
import openai
import chromadb

# fonte: https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/

openai.api_key = st.secrets.OPENAI_API_KEY

st.header("Interroga una base di conoscenza personalizzata")

if "messages" not in st.session_state.keys(): # Inizializza lo storico dei messaggi del chatbot
    st.session_state.messages = [
        {"role": "assistant", 
         "instructions": "Sei un ricercatore che aiuta gli utenti a trovare fonti storiche all'interno di una collezione di periodici. Rispondi sempre in italiano e citando sempre le fonti delle informazioni che restituisci.",
         "content": "Sono qui per aiutarti ad analizzare una digital library. Fammi una domanda e cercherò la risposta all'interno dei testi a mia disposizione, citando la fonte delle informazioni."}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner('Sto caricando e indicizzando i documenti. Potrebbe volerci qualche minuto'):
        # Rimuovi la vecchia collezione
        db = chromadb.PersistentClient(path="chroma_db")
        try:
            db.delete_collection(name='settegiorni_pipeline')
        except Exception as e:
            st.write("La collezione non esisteva, ne creo una nuova.")

        # Crea la collezione
        chroma_collection = db.get_or_create_collection('settegiorni_pipeline')
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Carica i documenti
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        documents = reader.load_data()

        # Crea l'indice
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=embed_model
        )

        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        return index
    
index = load_data()

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # here we can control how granular citation sources are, the default is 512
    citation_chunk_size=512,
)

if prompt := st.chat_input('La tua domanda'): # Chiede all'utente di fare una domanda
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Mostra il messaggio precedente
    with st.chat_message(message["role"]):
        st.write(message["content"])

# def extract_nodes(nodes):
#     extracted_nodes = []
#     for node in nodes:
#         file_name = node.metadata["file_name"]
#         text = node.get_text()
#         extracted_nodes.append((file_name, text))
#     return extracted_nodes
        

# Se l'ultimo messaggio non è dell'assistant, genera una nuova risposta
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Sto riflettendo..."):
            response = query_engine.query(prompt)
            # st.write(response.response)
            # nodes = response.source_nodes
            # extracted_nodes = extract_nodes(nodes)
            # for file_name, text in extracted_nodes:
            st.write(f'{response.response}')
            st.write('  \n\n')
            if response.source_nodes:
                st.write(f'Fonte: {response.source_nodes[0].get_text()}  \n {response.source_nodes[0].metadata}')
            else:
                st.write("Nessuna fonte trovata per questa risposta.")
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history