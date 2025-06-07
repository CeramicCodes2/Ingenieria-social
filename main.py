import poplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
import requests
from dotenv import load_dotenv
# dependences:
# pip install faiss-cpu sentence-transformers rank_bm25pip install faiss-cpu sentence-transformers rank_bm25
# --- Importaciones para Hybrid RAG ---
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Cargar variables de entorno desde .env
load_dotenv()

# --- Configuración del Bot ---
POP3_SERVER = os.getenv("POP3_SERVER")
POP3_PORT = int(os.getenv("POP3_PORT"))
POP3_USERNAME = os.getenv("POP3_USERNAME")
POP3_PASSWORD = os.getenv("POP3_PASSWORD")

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

LLM_STUDIO_URL = os.getenv("LLM_STUDIO_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

POLLING_INTERVAL_SECONDS = 60 # Cada cuánto tiempo el bot revisará nuevos correos

# --- Base de Conocimiento (Knowledge Base - KB) para RAG ---
# Puedes expandir esto cargando desde archivos, una base de datos, etc.
KNOWLEDGE_BASE = [
    "Nuestra empresa ofrece soporte técnico 24/7 a través de nuestro portal web y teléfono.",
    "Para restablecer tu contraseña, visita la página de inicio de sesión y haz clic en 'Olvidé mi contraseña'.",
    "Los horarios de atención al cliente son de Lunes a Viernes, de 9:00 AM a 6:00 PM (hora local).",
    "Nuestros servicios principales incluyen desarrollo de software, consultoría en la nube y capacitación en IA.",
    "Puedes encontrar nuestros tutoriales en video en el canal oficial de YouTube.",
    "El proceso de reembolso para productos comprados online tarda entre 3 y 5 días hábiles.",
    "Para problemas de facturación, por favor envía un correo a facturacion@nuestraempresa.com con tu número de pedido.",
    "Ofrecemos descuentos especiales para clientes empresariales con contratos anuales."
]

# --- Variables globales para RAG ---
rag_model = None
faiss_index = None
bm25_tokenizer = None
bm25_corpus = None
kb_documents = []

# --- Funciones de Utilidad ---

def initialize_rag_knowledge_base():
    """
    Inicializa el modelo de embedding, el índice FAISS y el índice BM25
    con la base de conocimiento.
    """
    global rag_model, faiss_index, bm25_tokenizer, bm25_corpus, kb_documents

    print("Inicializando la Base de Conocimiento para RAG...")
    rag_model = SentenceTransformer('all-MiniLM-L6-v2') # Modelo de embedding ligero y eficiente
    
    kb_documents = KNOWLEDGE_BASE # Usamos la lista directamente
    
    # 1. Preparar para búsqueda densa (FAISS)
    print("Creando embeddings de la KB...")
    document_embeddings = rag_model.encode(kb_documents, convert_to_tensor=True)
    document_embeddings_np = document_embeddings.cpu().numpy()

    d = document_embeddings_np.shape[1] # Dimensión de los embeddings
    faiss_index = faiss.IndexFlatL2(d) # Índice FAISS de distancia L2
    faiss_index.add(document_embeddings_np)
    print(f"Índice FAISS creado con {faiss_index.ntotal} documentos.")

    # 2. Preparar para búsqueda esparsa (BM25)
    print("Inicializando BM25...")
    tokenized_corpus = [doc.split(" ") for doc in kb_documents] # Tokenización simple por espacio
    bm25_tokenizer = lambda text: text.split(" ") # Función de tokenización para BM25
    bm25_corpus = BM25Okapi(tokenized_corpus) # Crear el objeto BM25

    print("Base de Conocimiento RAG inicializada.")

def hybrid_retrieve_documents(query_text, top_k_dense=2, top_k_sparse=2):
    """
    Realiza una búsqueda híbrida (densa y esparsa) en la base de conocimiento
    y devuelve los documentos más relevantes.
    """
    if rag_model is None or faiss_index is None or bm25_corpus is None:
        print("Error: La base de conocimiento RAG no ha sido inicializada.")
        return []

    print(f"Realizando búsqueda RAG para la consulta: '{query_text}'")

    # 1. Búsqueda Densa (FAISS)
    query_embedding = rag_model.encode(query_text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    D, I = faiss_index.search(query_embedding, top_k_dense)
    dense_results_indices = I[0]
    dense_documents = [kb_documents[idx] for idx in dense_results_indices if idx != -1]
    print(f"Resultados densos: {dense_documents}")

    # 2. Búsqueda Esparsa (BM25)
    tokenized_query = bm25_tokenizer(query_text)
    doc_scores = bm25_corpus.get_scores(tokenized_query)
    # Obtener índices de los top_k_sparse documentos con mayor puntuación
    sparse_results_indices = np.argsort(doc_scores)[::-1][:top_k_sparse]
    sparse_documents = [kb_documents[idx] for idx in sparse_results_indices if idx != -1]
    print(f"Resultados esparsos: {sparse_documents}")

    # 3. Combinar y Deduplicar resultados
    # Una estrategia simple: unir y eliminar duplicados manteniendo el orden de aparición.
    combined_documents = []
    seen = set()

    for doc in dense_documents:
        if doc not in seen:
            combined_documents.append(doc)
            seen.add(doc)
    for doc in sparse_documents:
        if doc not in seen:
            combined_documents.append(doc)
            seen.add(doc)

    print(f"Resultados RAG combinados: {combined_documents}")
    return combined_documents

def get_email_body(msg):
    """
    Extrae el cuerpo de texto plano de un objeto de mensaje de correo electrónico.
    Maneja mensajes multipart.
    """
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdisposition = str(part.get("Content-Disposition"))

            # Ignorar cualquier parte que no sea de texto plano y que sea un adjunto
            if ctype == "text/plain" and "attachment" not in cdisposition:
                return part.get_payload(decode=True).decode()
        return "" # No se encontró una parte de texto plano adecuada
    else:
        return msg.get_payload(decode=True).decode()

def generate_llm_response(prompt_text, retrieved_context=""):
    """
    Envía el texto del correo (y el contexto RAG) a LLM Studio y devuelve la respuesta.
    """
    headers = {"Content-Type": "application/json"}

    # Construir el prompt con el contexto recuperado
    if retrieved_context:
        full_prompt = (
            "Eres un asistente amable que responde a los correos electrónicos. "
            "Usa la siguiente información de contexto para responder a la pregunta del usuario. "
            "Si la información de contexto no es suficiente, responde con lo que sabes, pero prioriza el contexto si es relevante.\n\n"
            f"Contexto: {retrieved_context}\n\n"
            f"Pregunta del usuario (correo): {prompt_text}"
        )
    else:
        full_prompt = (
            "Eres un asistente amable que responde a los correos electrónicos. "
            "Sé conciso y útil.\n\n"
            f"Correo del usuario: {prompt_text}"
        )

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": full_prompt}
            # También podrías poner el correo del usuario como un rol 'user'
            # y el contexto como 'system' si el LLM lo interpreta mejor.
            # {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 200, # Aumentado para posibles respuestas más largas con contexto
        "temperature": 0.5 # Ajustado para respuestas más enfocadas al contexto
    }
    try:
        print(f"Enviando solicitud a LLM Studio: {LLM_STUDIO_URL}")
        response = requests.post(LLM_STUDIO_URL, headers=headers, json=payload, timeout=60) # Aumentado timeout
        response.raise_for_status() # Lanza una excepción para errores HTTP (4xx o 5xx)
        llm_response = response.json()
        print("Respuesta de LLM Studio recibida.")
        return llm_response['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con LLM Studio o al obtener respuesta: {e}")
        return "Lo siento, no pude generar una respuesta en este momento debido a un problema con mi cerebro de IA."

def send_reply_email(original_msg, llm_response_text):
    """
    Envía una respuesta al remitente original.
    """
    sender = SMTP_USERNAME
    recipient = original_msg['From']
    original_subject = original_msg['Subject']
    reply_subject = f"Re: {original_subject}"

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = reply_subject

    # Construir el cuerpo de la respuesta con el mensaje del LLM y el original
    reply_body = f"""Hola,

{llm_response_text}

---
Mensaje original:
De: {original_msg['From']}
Asunto: {original_msg['Subject']}
Fecha: {original_msg['Date']}

{get_email_body(original_msg)[:500]}... # Citar solo los primeros 500 caracteres del original
"""
    msg.attach(MIMEText(reply_body, 'plain'))

    try:
        print(f"Conectando a SMTP: {SMTP_SERVER}:{SMTP_PORT}")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls() # Usar TLS para seguridad
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(sender, recipient, msg.as_string())
        server.quit()
        print(f"Respuesta enviada a {recipient}")
    except smtplib.SMTPAuthenticationError:
        print("Error de autenticación SMTP. Revisa tu usuario y contraseña.")
    except smtplib.SMTPConnectError as e:
        print(f"Error de conexión SMTP: {e}. Revisa la URL del servidor y el puerto.")
    except Exception as e:
        print(f"Error al enviar correo: {e}")

def fetch_and_process_emails():
    """
    Conecta al servidor POP3, busca nuevos correos y los procesa.
    """
    try:
        print(f"Conectando a POP3: {POP3_SERVER}:{POP3_PORT}")
        # Usar POP3_SSL para conexiones seguras
        M = poplib.POP3_SSL(POP3_SERVER, POP3_PORT)
        M.user(POP3_USERNAME)
        M.pass_(POP3_PASSWORD)
        print("Conexión POP3 exitosa.")

        num_messages, total_size = M.stat()
        print(f"Tienes {num_messages} mensajes en tu buzón (total: {total_size} bytes).")

        if num_messages == 0:
            print("No hay nuevos correos para procesar.")
            M.quit()
            return

        for i in range(num_messages):
            print(f"Procesando mensaje {i+1}...")
            try:
                resp, lines, octets = M.retr(i + 1)
                msg_content = b'\r\n'.join(lines).decode('utf-8', errors='ignore')
                msg = email.message_from_string(msg_content)

                sender_email = msg['From']
                subject = msg['Subject']
                body = get_email_body(msg)

                print(f"De: {sender_email}")
                print(f"Asunto: {subject}")
                # print(f"Cuerpo: \n{body[:200]}...")

                # --- Paso de RAG: Recuperar contexto relevante ---
                retrieved_docs = hybrid_retrieve_documents(body)
                context_for_llm = "\n".join(retrieved_docs) # Unir documentos para el prompt

                print("Generando respuesta con LLM (con RAG)...")
                llm_response = generate_llm_response(body, context_for_llm)
                print(f"Respuesta del LLM: {llm_response}")

                send_reply_email(msg, llm_response)

                M.dele(i + 1)
                print(f"Mensaje {i+1} marcado para eliminación.")

            except Exception as e:
                print(f"Error al procesar el mensaje {i+1}: {e}")
                continue

        M.quit()
        print("Desconectado de POP3. Mensajes procesados y eliminados si se marcaron.")

    except poplib.error_proto as e:
        print(f"Error POP3: {e}. Revisa tu servidor, puerto, usuario y contraseña.")
    except Exception as e:
        print(f"Ocurrió un error inesperado al conectar o procesar correos: {e}")

# --- Bucle Principal del Bot ---
if __name__ == "__main__":
    # Inicializar la base de conocimiento RAG una vez al inicio
    initialize_rag_knowledge_base()

    print("Iniciando Email Bot con Hybrid RAG...")
    while True:
        fetch_and_process_emails()
        print(f"Esperando {POLLING_INTERVAL_SECONDS} segundos para la próxima revisión...")
        time.sleep(POLLING_INTERVAL_SECONDS)