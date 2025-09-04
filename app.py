# app.py
import streamlit as st
import os
import yaml
import bcrypt
import csv
import warnings
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- ИЗМЕНЕНО: Импорт FAISS ---
from langchain_community.vectorstores import FAISS
# --- КОНЕЦ ИЗМЕНЕНИЙ ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# Предполагается, что backup.py не использует Chroma. Если использует - его тоже нужно адаптировать.
from backup import create_backup, list_backups, restore_backup

# Подавляем предупреждения
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Загрузка переменных окружения
load_dotenv()

# --- ИЗМЕНЕНО: Константа для пути к индексу FAISS ---
PERSIST_DIRECTORY = "faiss_index"
# --- КОНЕЦ ИЗМЕНЕНИЙ ---

def get_llm():
    """Инициализирует и возвращает LLM через OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("❌ Ключ API OpenRouter (OPENROUTER_API_KEY) не найден.")
        st.stop()
    # Исправлен URL (убраны пробелы)
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1", # Исправлено
        temperature=0.3,
        max_tokens=8100,
    )
    return llm

class ModelManager:
    """Управление загрузкой и кэшированием моделей."""
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def preload_models():
        """Предзагрузка всех необходимых моделей при запуске"""
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings

class DocumentProcessor:
    """Обработка и индексация PDF-документов."""
    def __init__(self, embeddings):
        self.embeddings = embeddings
        # Для FAISS мы будем управлять индексом напрямую

    def determine_doc_type(self, pdf_path):
        """Определяет тип документа по имени файла"""
        filename = os.path.basename(pdf_path).lower()
        if 'годовой' in filename or 'отчет' in filename:
            return "Годовой отчет"
        elif 'памятка' in filename:
            return "Памятка"
        elif 'правила' in filename:
            return "Правила"
        elif 'положение' in filename:
            return "Положение"
        elif 'приказ' in filename:
            return "Приказ"
        elif 'форма' in filename or 'шаблон' in filename:
            return "Форма"
        else:
            return "Документ"

    def index_pdf(self, pdf_path):
        """Индексирует PDF-файл и сохраняет в FAISS."""
        if not os.path.exists(pdf_path):
            st.error(f"Файл {pdf_path} не найден!")
            return None
        with st.spinner(f"Индексирую {os.path.basename(pdf_path)}..."):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            if not documents:
                st.error(f"Не удалось извлечь содержимое из файла {pdf_path}")
                return None
            total_text = sum(len(doc.page_content.strip()) for doc in documents)
            if total_text == 0:
                st.error(f"Файл {pdf_path} не содержит извлекаемого текста (возможно сканированный PDF)")
                return None

            for doc in documents:
                if 'source' not in doc.metadata or not doc.metadata['source']:
                     doc.metadata['source'] = pdf_path
                # --- ИСПРАВЛЕНО: Опечатка в названии ключа ---
                if 'page' in doc.metadata:
                    doc.metadata['page'] = doc.metadata['page'] + 1
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                doc.metadata['doc_type'] = self.determine_doc_type(pdf_path)
                doc.metadata['filename'] = os.path.basename(pdf_path)
                content_lines = doc.page_content.strip().split('\n')
                if content_lines:
                    potential_title = content_lines[0].strip()
                    if len(potential_title) < 100 and potential_title.replace(" ", "").replace("-", "").isalnum():
                         doc.metadata['section_title'] = potential_title[:50]
                    else:
                         doc.metadata['section_title'] = "Без заголовка"
                else:
                     doc.metadata['section_title'] = "Без заголовка"

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=750,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            docs = text_splitter.split_documents(documents)
            if not docs:
                st.error(f"Не удалось создать чанки из файла {pdf_path}")
                return None

            # --- ИЗМЕНЕНО: Создание и сохранение FAISS индекса ---
            try:
                # Загружаем существующий индекс, если он есть
                if os.path.exists(PERSIST_DIRECTORY):
                    db = FAISS.load_local(PERSIST_DIRECTORY, self.embeddings, allow_dangerous_deserialization=True)
                    # Добавляем новые документы
                    db.add_documents(docs)
                else:
                    # Создаем новый индекс
                    db = FAISS.from_documents(docs, self.embeddings)
                
                # Сохраняем индекс
                db.save_local(PERSIST_DIRECTORY)
                st.success(f"✅ Документ {os.path.basename(pdf_path)} успешно индексирован!")
                return db
            except Exception as e:
                st.error(f"Ошибка при работе с FAISS: {e}")
                return None
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    def get_indexed_files(self):
        """Получает список индексированных файлов."""
        try:
            # --- ИЗМЕНЕНО: Проверка и загрузка из FAISS ---
            if os.path.exists(PERSIST_DIRECTORY):
                # FAISS не хранит метаданные о файлах напрямую.
                # Мы можем попробовать загрузить индекс и посмотреть примеры.
                # Но проще и надежнее хранить список файлов отдельно.
                # Создадим вспомогательный файл для отслеживания.
                tracker_file = os.path.join(PERSIST_DIRECTORY, "indexed_files.yaml")
                if os.path.exists(tracker_file):
                    with open(tracker_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        return data.get("files", [])
            return []
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        except Exception as e:
            st.warning(f"Не удалось получить список документов: {e}")
            return []

    def remove_document_from_index(self, filename):
        """Удаляет документ из индекса."""
        st.warning("Удаление отдельных документов из FAISS-индекса не поддерживается напрямую в этом приложении.")
        st.info("Вы можете удалить весь индекс и переиндексировать оставшиеся файлы.")
        return False
        # Примечание: FAISS не имеет встроенной функции удаления документов по метаданным.
        # Это требует более сложной логики, например, пересоздания индекса без ненужных документов.
        # Для простоты мы отключаем эту функцию.

    def _update_index_tracker(self, added_files=None, removed_files=None):
        """Вспомогательная функция для обновления списка индексированных файлов."""
        tracker_file = os.path.join(PERSIST_DIRECTORY, "indexed_files.yaml")
        data = {"files": []}
        if os.path.exists(tracker_file):
            with open(tracker_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {"files": []}
        
        current_files = set(data["files"])
        if added_files:
            current_files.update(added_files)
        if removed_files:
            current_files.difference_update(removed_files)
        
        data["files"] = sorted(list(current_files))
        
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        with open(tracker_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)

    def auto_index_all_pdfs(self):
        """Автоматически индексирует все PDF-файлы в папке data при запуске."""
        if not os.path.exists("data"):
            os.makedirs("data")
            st.info("Создана папка 'data'. Поместите сюда PDF-файлы для индексации.")
            return
        
        pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
        if not pdf_files:
            st.info("В папке 'data' не найдено PDF-файлов для индексации")
            return
        
        indexed_files = self.get_indexed_files()
        files_to_index = [f for f in pdf_files if f not in indexed_files]

        if files_to_index:
             st.info(f"Найдено {len(files_to_index)} новых файлов для индексации.")
             indexed_successfully = []
             for pdf_file in files_to_index:
                 pdf_path = f"data/{pdf_file}"
                 try:
                     # index_pdf теперь обновляет существующий индекс
                     if self.index_pdf(pdf_path) is not None:
                         indexed_successfully.append(pdf_file)
                 except Exception as e:
                     st.error(f"Ошибка при индексации {pdf_file}: {str(e)}")
             
             if indexed_successfully:
                 self._update_index_tracker(added_files=indexed_successfully)
                 st.success("✅ Автоматическая индексация завершена!")
             else:
                 st.warning("Индексация не добавила новых документов.")
        else:
             st.info("Все PDF-файлы в папке 'data' уже проиндексированы.")

class CorrectionManager:
    """Управление правками и шаблонами ответов."""
    def __init__(self):
        pass

    def load_corrections(self):
        """Загружает правки из YAML файла."""
        try:
            with open("corrections.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or []
        except FileNotFoundError:
            return []

    def save_correction(self, question, answer, sources):
        """Сохраняет правку в YAML файл."""
        corrections = self.load_corrections()
        new_correction = {
            "id": len(corrections) + 1,
            "question": question,
            "answer": answer,
            "sources": sources,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "author": st.session_state.get("username", "Пользователь")
        }
        corrections.append(new_correction)
        with open("corrections.yaml", "w", encoding="utf-8") as f:
            yaml.dump(corrections, f, allow_unicode=True, sort_keys=False)
        st.success("✅ Правка сохранена!")

class LogManager:
    """Логирование запросов и ответов."""
    @staticmethod
    def log_request(question, answer, sources, from_template=False):
        """Логирует запрос в CSV файл."""
        if not os.path.exists("logs"):
            os.makedirs("logs")
        log_file = "logs/requests.csv"
        file_exists = os.path.exists(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Дата", "Пользователь", "Вопрос", "Ответ",
                    "Источники", "Из шаблона", "Длина вопроса", "Длина ответа"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                st.session_state.get("username", "Неизвестный"),
                question,
                answer,
                "; ".join(sources) if sources else "",
                "Да" if from_template else "Нет",
                len(question),
                len(answer)
            ])

class RAGSystem:
    """Основная система вопросов-ответов (RAG)."""
    def __init__(self):
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.embeddings = self.model_manager.preload_models()
        # --- ИЗМЕНЕНО: Передаем embeddings в DocumentProcessor ---
        self.document_processor = DocumentProcessor(self.embeddings)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
        self.correction_manager = CorrectionManager()
        self.log_manager = LogManager()

    def query_rag(self, question):
        """Задаёт вопрос и возвращает ответ + источник."""
        try:
            # --- ИЗМЕНЕНО: Проверка через FAISS ---
            if not os.path.exists(PERSIST_DIRECTORY):
                response = {
                    "answer": "❌ База знаний не найдена. Пожалуйста, загрузите документы.",
                    "sources": ["Система"],
                    "from_template": False
                }
                self.log_manager.log_request(question, response["answer"], response["sources"], False)
                return response

            # --- ИЗМЕНЕНО: Загрузка FAISS индекса ---
            try:
                db = FAISS.load_local(PERSIST_DIRECTORY, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error(f"❌ Не удалось загрузить FAISS индекс: {e}")
                response = {
                    "answer": f"❌ Ошибка при загрузке базы знаний: {str(e)}",
                    "sources": ["Система"],
                    "from_template": False
                }
                self.log_manager.log_request(question, response["answer"], response["sources"], False)
                return response
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            
            try:
                llm = get_llm()
            except Exception as e:
                st.error(f"❌ Не удалось инициализировать LLM: {e}")
                response = {
                    "answer": f"❌ Ошибка при инициализации LLM: {str(e)}",
                    "sources": ["Система"],
                    "from_template": False
                }
                self.log_manager.log_request(question, response["answer"], response["sources"], False)
                return response
            
            # --- ИЗМЕНЕНО: Настройка ретривера для FAISS ---
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 75, # Можно настроить
                    "score_threshold": 0.3 # Можно настроить
                }
            )
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

            system_prompt = self.prompt_manager.load_system_prompt()

            prompt_template = system_prompt

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )

            result = qa.invoke({"query": question})
            answer = result["result"].strip()
            sources = result["source_documents"]
            source_info = []
            for doc in sources:
                meta = doc.metadata
                source_file = os.path.basename(meta.get('source', 'N/A'))
                page_num = meta.get('page', 'N/A')
                source_info.append(f"{source_file}, стр. {page_num}")

            def is_answer_useful(answer_text: str) -> bool:
                useless_phrases = ["в документах не указано", "конкретная информация отсутствует", "не удалось найти прямое упоминание"]
                return not any(phrase in answer_text.lower() for phrase in useless_phrases)
            
            if not is_answer_useful(answer):
                answer = "К сожалению, точной информации не найдено. Рекомендую обратить внимание на разделы, касающиеся интеллектуальной собственности или отчетности."

            response = {
                "answer": answer,
                "sources": list(set(source_info)),
                "from_template": False
            }
            self.log_manager.log_request(question, response["answer"], response["sources"], False)
            return response
        except Exception as e:
            response = {
                "answer": f"❌ Ошибка при обработке запроса: {str(e)}",
                "sources": ["Система"],
                "from_template": False
            }
            self.log_manager.log_request(question, response["answer"], response["sources"], False)
            return response

class UserManager:
    """Управление пользователями и аутентификацией."""
    @staticmethod
    def load_users():
        """Загружает пользователей из YAML файла."""
        try:
            with open("users.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {"users": []}

    @staticmethod
    def save_users(users_data):
        """Сохраняет пользователей в YAML файл."""
        try:
            with open("users.yaml", "w", encoding="utf-8") as f:
                yaml.dump(users_data, f, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            st.error(f"Ошибка при сохранении пользователей: {e}")
            return False

    @staticmethod
    def authenticate_user(username, password):
        """Проверяет учетные данные пользователя."""
        users_data = UserManager.load_users()
        for user in users_data.get("users", []):
            if user["username"] == username:
                if bcrypt.checkpw(password.encode('utf-8'), user["password_hash"].encode('utf-8')):
                    return user
        return None

    @staticmethod
    def hash_password(password):
        """Хэширует пароль."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

class AdminPanel:
    """Админ-панель управления системой."""
    def __init__(self):
        self.user_manager = UserManager()
        self.correction_manager = CorrectionManager()
        self.rag_system = RAGSystem()

    def show_admin_panel(self):
        """Отображает админ-панель."""
        st.title("🔒 Админ-панель")
        admin_tabs = st.tabs(["Пользователи", "Журнал правок", "Статистика", "Резервное копирование"])
        
        with admin_tabs[0]:
            st.subheader("👥 Управление пользователями")
            users_data = self.user_manager.load_users()
            if users_data.get("users"):
                for i, user in enumerate(users_data["users"]):
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    with col1:
                        st.write(f"**{user['username']}**")
                    with col2:
                        st.write(user['role'])
                    with col3:
                        st.write("● Активен")
                    with col4:
                        pass
            st.divider()
            with st.form("add_user_form"):
                new_username = st.text_input("Логин")
                new_password = st.text_input("Пароль", type="password")
                new_role = st.selectbox("Роль", ["user", "editor", "admin"])
                submit_button = st.form_submit_button("Добавить пользователя")
                if submit_button:
                    if new_username and new_password:
                        users_data = self.user_manager.load_users()
                        if any(user["username"] == new_username for user in users_data.get("users", [])):
                            st.error(f"Пользователь с именем {new_username} уже существует!")
                        else:
                            hashed_password = self.user_manager.hash_password(new_password)
                            new_user = {
                                "username": new_username,
                                "password_hash": hashed_password,
                                "role": new_role
                            }
                            users_data.setdefault("users", []).append(new_user)
                            if self.user_manager.save_users(users_data):
                                st.success(f"Пользователь {new_username} успешно добавлен с ролью {new_role}!")
                                st.rerun()
                            else:
                                st.error("Не удалось сохранить пользователя.")
                    else:
                        st.error("Заполните все поля")
        
        with admin_tabs[1]:
            st.subheader("📝 Журнал правок")
            corrections = self.correction_manager.load_corrections()
            if corrections:
                correction_data = []
                for correction in corrections:
                    correction_data.append({
                        "ID": correction["id"],
                        "Вопрос": correction["question"][:50] + "..." if len(correction["question"]) > 50 else correction["question"],
                        "Автор": correction["author"],
                        "Дата": correction["date"]
                    })
                st.table(correction_data)
            else:
                st.info("Пока нет сохраненных правок")
        
        with admin_tabs[2]:
            st.subheader("📊 Статистика")
            log_file = "logs/requests.csv"
            if os.path.exists(log_file):
                import pandas as pd
                try:
                    df = pd.read_csv(log_file)
                    total_requests = len(df)
                    st.metric("Всего запросов", total_requests)
                    no_answer_requests = len(df[df['Ответ'].str.contains("❌|не найдена|Ошибка|К сожалению, точной информации не найдено", case=False, na=False)])
                    if total_requests > 0:
                        no_answer_percentage = (no_answer_requests / total_requests) * 100
                        st.metric("Запросов без ответа (%)", f"{no_answer_percentage:.1f}%")
                    st.subheader("Топ пользователей")
                    user_stats = df['Пользователь'].value_counts()
                    st.bar_chart(user_stats)
                    st.divider()
                    st.subheader("📥 Экспорт данных")
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Скачать логи (CSV)",
                        data=csv,
                        file_name=f"requests_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Ошибка при загрузке статистики: {e}")
            else:
                st.info("Логи запросов пока отсутствуют")
        
        with admin_tabs[3]:
            st.subheader("💾 Резервное копирование")
            if st.button("Создать резервную копию"):
                with st.spinner("Создаем резервную копию..."):
                    backup_file = create_backup()
                    if backup_file:
                        st.success(f"✅ Резервная копия создана: {os.path.basename(backup_file)}")
                    else:
                        st.error("❌ Не удалось создать резервную копию")
            st.divider()
            st.subheader("Список резервных копий")
            backups = list_backups()
            if backups:
                for backup in backups:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(backup)
                    with col2:
                        if st.button("Восстановить", key=f"restore_{backup}"):
                            with st.spinner("Восстанавливаем из резервной копии..."):
                                if restore_backup(os.path.join("backup", backup)):
                                    st.success("✅ Восстановление выполнено успешно!")
                                    st.info("Пожалуйста, перезапустите приложение")
                                else:
                                    st.error("❌ Не удалось восстановить из резервной копии")
            else:
                st.info("Резервные копии не найдены")

class SkolkovoConsultantApp:
    """Главное приложение Streamlit."""
    def __init__(self):
        self.rag_system = RAGSystem()
        self.admin_panel = AdminPanel()
        self.user_manager = UserManager()

    def main(self):
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

        if not st.session_state.authenticated:
            st.title("🏢 Консультант по проекту «Сколково»")
            st.subheader("🔐 Вход в систему")
            with st.form("login_form"):
                username = st.text_input("Логин")
                password = st.text_input("Пароль", type="password")
                login_button = st.form_submit_button("Войти")
                if login_button:
                    user = self.user_manager.authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.username = user["username"]
                        st.session_state.role = user["role"]
                        st.success(f"Добро пожаловать, {username}!")
                        st.rerun()
                    else:
                        st.error("Неверный логин или пароль")
            return

        if st.session_state.role == "user":
            st.set_page_config(page_title="Консультант по Сколково", page_icon="🏢", layout="centered")
        else:
            st.set_page_config(page_title="Консультант по Сколково", page_icon="🏢", layout="wide")

        if st.session_state.role == "admin":
            page = st.sidebar.selectbox("Навигация", ["Чат", "Админ-панель"])
        else:
            page = "Чат"

        if st.session_state.role == "user":
            st.title("🏢 Консультант по проекту «Сколково»")
            st.caption(f"Вы вошли как: {st.session_state.username} ({st.session_state.role})")
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("🚪 Выйти"):
                    st.session_state.authenticated = False
                    st.session_state.username = None
                    st.session_state.role = None
                    messages_key = f"messages_{st.session_state.get('username', 'default')}"
                    if messages_key in st.session_state:
                        del st.session_state[messages_key]
                    st.rerun()
        else:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.title("🏢 Консультант по проекту «Сколково»")
            with col2:
                if st.button("🚪 Выйти"):
                    st.session_state.authenticated = False
                    st.session_state.username = None
                    st.session_state.role = None
                    messages_key = f"messages_{st.session_state.get('username', 'default')}"
                    if messages_key in st.session_state:
                        del st.session_state[messages_key]
                    st.rerun()
            st.caption(f"Вы вошли как: {st.session_state.username} ({st.session_state.role})")

        if page == "Админ-панель":
            self.admin_panel.show_admin_panel()
            return

        messages_key = f"messages_{st.session_state.username}"
        if messages_key not in st.session_state:
            st.session_state[messages_key] = []

        editing_key = f"editing_message_index_{st.session_state.username}"
        edit_question_key = f"edit_question_{st.session_state.username}"
        edit_answer_key = f"edit_answer_{st.session_state.username}"
        edit_sources_key = f"edit_sources_{st.session_state.username}"
        
        if editing_key not in st.session_state:
            st.session_state[editing_key] = None
        if edit_question_key not in st.session_state:
            st.session_state[edit_question_key] = ""
        if edit_answer_key not in st.session_state:
            st.session_state[edit_answer_key] = ""
        if edit_sources_key not in st.session_state:
            st.session_state[edit_sources_key] = []

        if "models_loaded" not in st.session_state:
            st.session_state.embeddings = ModelManager.preload_models()
            st.session_state.models_loaded = True

        # --- ИЗМЕНЕНО: Проверяем существование индекса на диске ---
        # Вместо проверки st.session_state["indexed_on_startup"]
        if not os.path.exists(PERSIST_DIRECTORY): # PERSIST_DIRECTORY = "faiss_index"
            with st.spinner("Индексация документов..."):
                # --- ИЗМЕНЕНО: Используем метод из rag_system ---
                self.rag_system.document_processor.auto_index_all_pdfs()
                # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            # st.session_state.indexed_on_startup = True # Эту строку можно удалить
        else:
            # Добавляем сообщение в сайдбар, если индекс уже существует
            st.sidebar.success("✅ База знаний уже загружена.")
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state[messages_key]):
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        st.markdown(message["content"])
                        if "sources" in message and message["sources"]:
                            with st.expander("Источники", expanded=False):
                                for source in message["sources"]:
                                    st.write(source)
                        if st.session_state[editing_key] == i:
                            st.subheader("Редактирование ответа")
                            edited_answer = st.text_area("Отредактированный ответ:", value=st.session_state[edit_answer_key], height=150)
                            edited_sources = st.text_input("Источники (через запятую):", value=", ".join(st.session_state[edit_sources_key]))
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("💾 Сохранить правку", key=f"save_{i}"):
                                    sources_list = [s.strip() for s in edited_sources.split(",") if s.strip()]
                                    self.rag_system.correction_manager.save_correction(
                                        st.session_state[edit_question_key],
                                        edited_answer,
                                        sources_list
                                    )
                                    st.session_state[messages_key][i]["content"] = edited_answer
                                    st.session_state[messages_key][i]["sources"] = sources_list
                                    st.session_state[editing_key] = None
                                    st.session_state[edit_question_key] = ""
                                    st.session_state[edit_answer_key] = ""
                                    st.session_state[edit_sources_key] = []
                                    st.rerun()
                            with col2:
                                if st.button("❌ Отмена", key=f"cancel_{i}"):
                                    st.session_state[editing_key] = None
                                    st.session_state[edit_question_key] = ""
                                    st.session_state[edit_answer_key] = ""
                                    st.session_state[edit_sources_key] = []
                                    st.rerun()
                            with col3:
                                st.info(f"ID шаблона: {message.get('template_id', 'N/A')}" if message.get("from_template") else "RAG ответ")
                        else:
                            if st.session_state.role in ["editor", "admin"]:
                                if st.button("✏️ Отредактировать ответ", key=f"edit_btn_{i}_{hash(str(message.get('content', '')))}"):
                                    st.session_state[editing_key] = i
                                    st.session_state[edit_question_key] = message.get("question", "")
                                    st.session_state[edit_answer_key] = message["content"]
                                    st.session_state[edit_sources_key] = message.get("sources", [])
                                    st.rerun()
                    else:
                        st.markdown(message["content"])

        if prompt := st.chat_input("Введите ваш вопрос..."):
            st.session_state[messages_key].append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Думаю..."):
                        response = self.rag_system.query_rag(prompt)
                    st.markdown(response["answer"])
                    if response["sources"] and response["sources"] != ["Система"]:
                        with st.expander("Источники", expanded=True):
                            for source in response["sources"]:
                                st.write(source)
                    st.session_state[messages_key].append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                        "question": prompt,
                        "from_template": response["from_template"],
                        "template_id": response.get("template_id")
                    })
            st.rerun()

        if st.session_state.role in ["admin", "editor"]:
            with st.sidebar:
                st.header("⚙️ Управление")
                # --- ИЗМЕНЕНО: Проверка через FAISS ---
                if os.path.exists(PERSIST_DIRECTORY):
                    st.success("✅ База знаний загружена")
                    # --- ИЗМЕНЕНО: Используем метод из rag_system ---
                    indexed_files = self.rag_system.document_processor.get_indexed_files()
                    # --- КОНЕЦ ИЗМЕНЕНИЙ ---
                    if indexed_files:
                        st.subheader("Индексированные документы:")
                        for filename in indexed_files:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📄 {filename}")
                            with col2:
                                # --- ИЗМЕНЕНО: Отключена кнопка удаления ---
                                # if st.session_state.role == "admin":
                                #     if st.button("❌", key=f"delete_{filename}", help=f"Удалить {filename}"):
                                #         if self.rag_system.document_processor.remove_document_from_index(filename):
                                #             st.rerun()
                                # else:
                                st.write("(Удаление отдельных документов из FAISS не поддерживается)")
                                # --- КОНЕЦ ИЗМЕНЕНИЙ ---
                else:
                    st.warning("❌ База знаний не найдена")
                
                if st.session_state.role == "admin":
                    st.divider()
                    st.subheader("Добавить документ")
                    uploaded_file = st.file_uploader("Загрузите PDF", type="pdf", key="uploader")
                    if uploaded_file is not None:
                        file_path = f"data/{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.success(f"✅ Файл {uploaded_file.name} сохранен в папке data/")
                        if st.button("Индексировать документ", key="index_btn"):
                            # --- ИЗМЕНЕНО: Используем метод из rag_system ---
                            self.rag_system.document_processor.index_pdf(file_path)
                            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
                            st.rerun()
                
                if st.session_state.role == "admin":
                    st.divider()
                    if st.button("🔄 Обновить индекс"):
                        with st.spinner("Обновляем индекс..."):
                            # --- ИЗМЕНЕНО: Используем метод из rag_system ---
                            self.rag_system.document_processor.auto_index_all_pdfs()
                            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
                        st.success("Индекс обновлен!")
                        st.rerun()
                
                st.divider()
                st.subheader("Сохраненные правки")
                corrections = self.rag_system.correction_manager.load_corrections()
                if corrections:
                    for correction in corrections[-5:]:
                        with st.expander(f"Вопрос: {correction['question'][:50]}...", expanded=False):
                            st.write(f"**Вопрос:** {correction['question']}")
                            st.write(f"**Ответ:** {correction['answer']}")
                            st.write(f"**Источники:** {', '.join(correction['sources'])}")
                            st.write(f"**Дата:** {correction['date']}")
                            st.write(f"**Автор:** {correction['author']}")
                            st.write(f"**ID:** {correction['id']}")
                else:
                    st.info("Пока нет сохраненных правок")

class PromptManager:
    """Управление системными промптами."""
    @staticmethod
    def load_system_prompt():
        """Загружает системный промпт из файла."""
        try:
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return """Ты — эксперт по нормативным документам и льготам участников проекта «Сколково».
Твоя задача — предоставить точный, структурированный и краткий ответ на вопрос, используя исключительно информацию из предоставленного контекста.

### Стиль ответа:
- Ответ должен быть профессиональным, лаконичным и структурированным, как в юридической справке.
- Не добавляй информацию, которой нет в контексте.
- Используй маркированные списки, заголовки и подзаголовки для структурирования.

### Правила:
1. **Если вопрос касается сроков**:
   - Укажи точные даты или периоды, если это возможно по контексту.
   - Если точные сроки не указаны в контексте, укажи, что информация отсутствует.

2. **Если вопрос о санкциях или последствиях**:
   - Укажи, что будет при нарушении условий и при каких условиях утрачиваются льготы или статусы.
   - Если последствия не указаны в контексте, сообщи об этом.

3. **Если вопрос о документах**:
   - Перечисли все необходимые документы и условия их подачи.
   - Если документы не упомянуты в контексте, сообщи, что информация отсутствует.

4. **Если вопрос о льготах**:
   - Укажи льготу, ее суть, ограничения и условия утраты.
   - Если льгота не упомянута в контексте, сообщи, что информация отсутствует.

5. **Если вопрос о проверках**:
   - Опиши, как осуществляется проверка, какие критерии и сроки применяются.
   - Если методика проверки не указана, сообщи об этом.

6. **Если вопрос о деятельности**:
   - Укажи, что разрешено, а что запрещено.
   - Если условия деятельности не упомянуты в контексте, сообщи об этом.

7. **Указывай нормативные акты**:
   - Включай ссылки на все **нормативные акты**, если они упомянуты в контексте.
   
8. **Формулировка ответа:**
   - Ответ должен быть максимально кратким и содержать только прямой ответ на вопрос.
   - Не приводите длинные списки правил или объяснений, если вопрос не требует этого.
   - Если вопрос касается конкретной даты, события или факта, ответ должен начинаться с этого факта.


### Примеры:
Пример 1:
Вопрос: Какие налоги не платит компания-резидент «Сколково»?
Ответ:
Компания-резидент «Сколково» освобождается от ряда налогов, если соблюдаются условия статуса и лимитов.

1. **Налог на прибыль организаций**
   - **Суть льготы:** Полное освобождение на срок 10 лет с даты получения статуса резидента (п. 1 ст. 246.1 НК РФ).
   - **По истечении 10 лет** право прекращается, даже при повторном получении статуса.

---

Пример 2:
Вопрос: До какого числа нужно подать годовой отчет за 2024 год в Фонд «Сколково»?
Ответ:
Годовой отчет за 2024 год должен быть подан не позднее **7 апреля 2025 года**.

Контекст:
{context}

Вопрос:
{question}

Ответ (официально, ясно, по делу):"""

if __name__ == "__main__":
    app = SkolkovoConsultantApp()
    app.main()

