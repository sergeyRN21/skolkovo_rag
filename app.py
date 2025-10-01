import streamlit as st
import os
import yaml
import json
import bcrypt
import csv
import warnings
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA # Удалено, так как используем Runnable
from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate # Заменено
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from backup import create_backup, list_backups, restore_backup

# Подавляем предупреждения
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Загрузка переменных окружения
load_dotenv()

# ===========================
# 🧩 КОНФИГУРАЦИЯ (ВСТРОЕННАЯ)
# ===========================
CONFIG = {
    "app": {
        "name": "Консультант по Сколково",
        "page_icon": "🏢"
    },
    "llm": {
        "model": "google/gemini-2.5-flash",
        "temperature": 0.2,
        "max_tokens": 10000,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY")
    },
    "chroma": {
        "persist_directory": "chroma_skolkovo"
    },
    # Уточнение параметров поиска для повышения точности
    "retriever": {
        "k": 45,  # Уменьшено для большей точности
        "score_threshold": 0.3 # Увеличено для релевантности
    }
}

def get_llm():
    """Инициализирует и возвращает LLM через OpenRouter API."""
    api_key = CONFIG["llm"]["api_key"]
    if not api_key:
        st.error("❌ Ключ API OpenRouter (OPENROUTER_API_KEY) не найден.")
        st.stop()
    llm = ChatOpenAI(
        model=CONFIG["llm"]["model"],
        openai_api_key=api_key,
        openai_api_base=CONFIG["llm"]["base_url"],
        temperature=CONFIG["llm"]["temperature"],
        max_tokens=CONFIG["llm"]["max_tokens"],
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
        """Индексирует PDF-файл и сохраняет в Chroma."""
        if not os.path.exists(pdf_path):
            st.error(f"Файл {pdf_path} не найден!")
            return None

        with st.spinner():
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
                if 'page' in doc.metadata:
                    doc.metadata['page'] = doc.metadata['page'] + 1
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
                chunk_size=512,
                chunk_overlap=128,
                separators=[
                    "\n\n\n",  # Три переноса — вероятно, раздел
                    "\n\n",    # Два переноса — подраздел
                    "\n•", "\n- ",  # Маркированные списки
                    "\n", ". ", " ", ""
                ]
            )
            docs = text_splitter.split_documents(documents)
            if not docs:
                st.error(f"Не удалось создать чанки из файла {pdf_path}")
                return None

            db = Chroma.from_documents(docs, self.embeddings, persist_directory=CONFIG["chroma"]["persist_directory"])
            return db

    def get_indexed_files(self):
        """Получает список индексированных файлов."""
        try:
            if os.path.exists(CONFIG["chroma"]["persist_directory"]):
                db = Chroma(persist_directory=CONFIG["chroma"]["persist_directory"], embedding_function=self.embeddings)
                docs = db.get()
                if docs and docs.get('ids'):
                    sources = set()
                    metadatas = docs.get('metadatas', [])
                    if metadatas:
                        for metadata in metadatas:
                            source_path = metadata.get('source')
                            if source_path:
                                filename = os.path.basename(source_path)
                                sources.add(filename)
                        return list(sources)
            return []
        except Exception as e:
            st.warning(f"Не удалось получить список документов: {e}")
            return []

    def remove_document_from_index(self, filename):
        """Удаляет документ из индекса."""
        try:
            if os.path.exists(CONFIG["chroma"]["persist_directory"]):
                db = Chroma(persist_directory=CONFIG["chroma"]["persist_directory"], embedding_function=self.embeddings)
                docs = db.get()
                ids_to_delete = []
                for i, metadata in enumerate(docs['metadatas']):
                    if 'source' in metadata and os.path.basename(metadata['source']) == filename:
                        ids_to_delete.append(docs['ids'][i])
                if ids_to_delete:
                    db.delete(ids_to_delete)
                    st.success(f"✅ Документ {filename} удален из индекса!")
                    return True
                else:
                    st.warning(f"Документ {filename} не найден в индексе")
                    return False
        except Exception as e:
            st.error(f"Ошибка при удалении документа: {e}")
            return False

    def auto_index_all_pdfs(self):
        """Автоматически индексирует все PDF-файлы в папке data при запуске."""
        if not os.path.exists("data"):
            os.makedirs("data")
            return

        pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
        if not pdf_files:
            st.warning("В папке 'data' не найдено PDF-файлов для индексации")
            return

        indexed_files = self.get_indexed_files()
        files_to_index = []
        for pdf_file in pdf_files:
            if pdf_file not in indexed_files:
                files_to_index.append(pdf_file)

        if files_to_index:
            for pdf_file in files_to_index:
                pdf_path = f"data/{pdf_file}"
                try:
                    self.index_pdf(pdf_path)
                except Exception as e:
                    st.error(f"Ошибка при индексации {pdf_file}: {str(e)}")

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
        """Сохраняет правку в YAML файл и как шаблон."""
        # 1. Сохраняем в corrections.yaml (для истории)
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

        # 2. Добавляем в templates.yaml как шаблон
        try:
            with open("templates.yaml", "r", encoding="utf-8") as f:
                templates = yaml.safe_load(f) or []
        except FileNotFoundError:
            templates = []

        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Создаем ПРОСТОЙ паттерн с границами слов
        # 1. Экранируем специальные символы регулярного выражения в вопросе
        escaped_question = re.escape(question.strip())
        # 2. Добавляем границы слова в начале и конце
        #    \b гарантирует, что совпадение будет с целым словом/фразой
        simple_pattern = rf"{escaped_question}"
        
        new_template = {
            "id": len(templates) + 1,
            "question_pattern": simple_pattern, # Используем простой паттерн
            "answer": answer
        }
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        templates.append(new_template)
        with open("templates.yaml", "w", encoding="utf-8") as f:
            yaml.dump(templates, f, allow_unicode=True, sort_keys=False)

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

# ===========================
# 🧠 ЯДРО RAG СИСТЕМЫ С ДОРАБОТКАМИ
# ===========================
class RAGSystem:
    """Основная система вопросов-ответов (RAG) с улучшениями."""
    def __init__(self):
        self.model_manager = ModelManager()
        self.embeddings = self.model_manager.preload_models()
        self.document_processor = DocumentProcessor(self.embeddings)
        self.correction_manager = CorrectionManager()
        self.log_manager = LogManager()
        # Загружаем шаблоны и базу ссылок при инициализации
        self.templates = self.load_templates()
        self.legal_db = self.load_legal_db()

    def load_templates(self):
        """Загружает шаблоны ответов из YAML файла и сортирует по специфичности."""
        try:
            with open("templates.yaml", "r", encoding="utf-8") as f:
                templates = yaml.safe_load(f) or []
            # Сортировка по длине паттерна по убыванию (более длинные/специфичные первые)
            templates.sort(key=lambda t: len(t.get("question_pattern", "")), reverse=True)
            # Отладка (убрать после проверки)
            # print("Загруженные и отсортированные шаблоны:")
            # for t in templates:
            #     print(f"  - ID: {t['id']}, Pattern: {t['question_pattern']}")
            return templates
        except FileNotFoundError:
            return [] # Если файла нет, возвращаем пустой список

    def load_legal_db(self):
        """Загружает базу нормативных актов для валидации ссылок."""
        try:
            with open("legal_docs_db.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # Создаем дефолтную базу
            default_legal_db = {
                "337-Пр": {
                    "full_name": "Приказ Фонда «Сколково» от 26 декабря 2024 г. № 337-Пр «Об утверждении Порядка представления в Фонд «Сколково» годовых отчетов участников проекта «Сколково» за 2024 год»"
                },
                "292-Пр": {
                    "full_name": "Приказ Фонда «Сколково» от 30.11.2018 № 292-Пр «Об утверждении Методики подтверждения существенной степени локализации...»"
                }
            }
            with open("legal_docs_db.json", "w", encoding="utf-8") as f:
                json.dump(default_legal_db, f, ensure_ascii=False, indent=2)
            return default_legal_db
    def classify_question(self, question: str) -> str:
        """
        Классифицирует вопрос на один из предопределенных типов.
        
        Returns:
            str: Тип вопроса (например, 'deadline', 'penalty', 'benefit', 'document', 'activity', 'verification', 'general')
        """
        q_lower = question.lower().strip()

        # --- Определение типа вопроса по ключевым словам ---
        
        # 1. Сроки / Дедлайны
        if any(keyword in q_lower for keyword in ["до какого числа", "когда нужно", "срок сдачи", "дедлайн"]):
            return "deadline"

        # 2. Последствия / Санкции
        # Обратите внимание: "что будет если" и "что если опоздать" более специфичны, чем просто "что будет"
        if any(keyword in q_lower for keyword in ["что будет если", "что если опоздать", "последствия", "санкци", "лишени", "исключен"]):
            return "penalty"

        # 3. Льготы / Налоги
        if any(keyword in q_lower for keyword in ["льгот", "не платит", "освобожд", "налог", "взнос"]):
            return "benefit"

        # 4. Документы
        if any(keyword in q_lower for keyword in ["документ", "прилож", "форма", "анализ счетов", "осв"]): # "осв" для ОСВ
            return "document"

        # 5. Деятельность / Правила
        if any(keyword in q_lower for keyword in ["может ли", "можно ли", "считаться", "субаренд", "профильн", "непрофильн"]):
            return "activity"

        # 6. Проверки / Критерии
        if any(keyword in q_lower for keyword in ["как фонд провер", "что значит", "критер", "порог", "локализац"]):
            return "verification"

        # 7. По умолчанию - Общий тип
        return "general"
    def match_template(self, question: str):
        """Проверяет, соответствует ли вопрос одному из шаблонов."""
        # Отладка (убрать после проверки)
        print(f"Проверка шаблонов для вопроса: '{question}'")
        for template in self.templates:
            try:
                if re.search(template["question_pattern"], question, re.IGNORECASE):
                    # Отладка (убрать после проверки)
                    print(f"  Совпал шаблон ID {template['id']}: '{template['question_pattern']}'")
                    return template
            except re.error as e:
                # Отладка (убрать после проверки)
                print(f"  Ошибка в регулярном выражении шаблона ID {template['id']}: {e}")
                st.warning(f"Ошибка в шаблоне ID {template['id']}: некорректное регулярное выражение.")
        # Отладка (убрать после проверки)
        print("  Совпадений не найдено")
        return None

# ... (внутри class RAGSystem)
    def post_process_answer(self, answer: str) -> str:
        """Пост-обработка ответа: исправление грамматики, удаление дублей, очистка."""
        # 1. Базовые грамматические замены
        replacements = {
            "ложных информации": "ложной информации",
            "документов, приложенных к отчету.": "документов, приложенных к отчету;",
            "за исключением случая, предусмотренного подпунктом «а» настоящего пункта)": "за исключением случая, предусмотренного подпунктом «а»)",
            "[a]": "", "[b]": "", "[c]": "", "[d]": "", "[e]": "", "[f]": "",
            "[g]": "", "[h]": "", "[i]": "", "[j]": "", "[k]": "", "[l]": "",
            "[m]": "", "[n]": ""
        }
        for old, new in replacements.items():
            answer = answer.replace(old, new)

        # 2. Удаление дублирующих строк (упрощенно)
        lines = answer.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                unique_lines.append(line)
                seen.add(stripped)
        answer = '\n'.join(unique_lines)

        # 3. Валидация и расширение нормативных ссылок
        for short_code, details in self.legal_db.items():
            # Ищем шаблон "Приказ ... № XXX-Пр"
            pattern = rf"(Приказ.*?№\s*{re.escape(short_code)})"
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                answer = re.sub(pattern, details["full_name"], answer, flags=re.IGNORECASE)

        return answer.strip()

    def query_rag(self, question):
        """Задаёт вопрос и возвращает ответ + источник."""
        try:
            # --- ШАГ 1: Проверка шаблонов ---
            matched_template = self.match_template(question)
            if matched_template:
                response = {
                    "answer": matched_template["answer"],
                    "sources": ["Шаблон"],
                    "from_template": True,
                    "template_id": matched_template["id"]
                }
                self.log_manager.log_request(question, response["answer"], response["sources"], True)
                return response

            # --- ШАГ 2: Проверка наличия базы знаний ---
            if not os.path.exists(CONFIG["chroma"]["persist_directory"]):
                response = {
                    "answer": "❌ База знаний не найдена. Пожалуйста, загрузите документы.",
                    "sources": ["Система"],
                    "from_template": False
                }
                self.log_manager.log_request(question, response["answer"], response["sources"], False)
                return response

            # --- ШАГ 3: Инициализация LLM и Retriever ---
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

            # --- ШАГ 3: Инициализация LLM и Retriever с адаптивными параметрами ---
            # ... (инициализация llm) ...
            db = Chroma(persist_directory=CONFIG["chroma"]["persist_directory"], embedding_function=self.embeddings)
            
            # 1. Классифицируем вопрос
            question_type = self.classify_question(question)
            # Отладка (можно убрать позже)
            # print(f"[DEBUG] Классифицированный тип вопроса: '{question_type}' для вопроса: '{question}'")

            # 2. Определяем параметры поиска на основе типа вопроса
            retrieval_params = {
                "k": CONFIG["retriever"]["k"], # Значения по умолчанию
                "score_threshold": CONFIG["retriever"]["score_threshold"]
            }

            if question_type == "deadline":
                # Для сроков: высокая точность, меньше результатов
                retrieval_params["k"] = 45
                retrieval_params["score_threshold"] = 0.55
            elif question_type == "penalty":
                # Для последствий: широкий охват, найти все возможные санкции
                retrieval_params["k"] = 30
                retrieval_params["score_threshold"] = 0.4
            elif question_type == "benefit":
                 # Для льгот: баланс точности и полноты
                retrieval_params["k"] = 45
                retrieval_params["score_threshold"] = 0.3
            elif question_type == "document":
                # Для документов: высокая точность, конкретные списки
                retrieval_params["k"] = 45
                retrieval_params["score_threshold"] = 0.3
            elif question_type == "activity":
                # Для деятельности: средний баланс
                retrieval_params["k"] = 45
                retrieval_params["score_threshold"] = 0.2
            elif question_type == "verification":
                # Для проверок: средний баланс, возможно чуть шире
                retrieval_params["k"] = 40
                retrieval_params["score_threshold"] = 0.38
            # Для 'general' используем значения по умолчанию
            
            # Отладка (можно убрать позже)
            # print(f"[DEBUG] Параметры поиска: {retrieval_params}")

            # 3. Создаем retriever с адаптивными параметрами
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=retrieval_params
            )

            # --- ШАГ 4: Создание новой цепочки с ChatPromptTemplate ---
            # Функция для форматирования документов
            def format_docs(docs):
                return "\n\n---\n\n".join(doc.page_content for doc in docs)

            # Функция для извлечения источников (для логирования)
            def retrieve_and_format_sources(inputs: dict):
                docs = retriever.get_relevant_documents(inputs["question"])
                source_info = []
                for doc in docs:
                    meta = doc.metadata
                    source_file = os.path.basename(meta.get('source', 'N/A'))
                    page_num = meta.get('page', 'N/A')
                    source_info.append(f"{source_file}, стр. {page_num}")
                return {
                    "context": format_docs(docs),
                    "question": inputs["question"],
                    "source_info": list(set(source_info)),
                    "source_documents": docs # Для возможной дальнейшей обработки
                }

            # Улучшенный промпт с ChatPromptTemplate и примерами
            improved_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Ты — эксперт по нормативным документам и льготам участников проекта «Сколково».
Твоя задача — предоставить точный, структурированный и ПОЛНЫЙ ответ на вопрос, используя исключительно информацию из предоставленного контекста.

### Стиль ответа:
- Отвечай ЧЕТКО, структурированно и ПОЛНО. Не упускай важные детали и условия, даже если это сделает ответ немного длиннее. Краткость важна, но не в ущерб информативности.
- ❗ Отвечай ТОЛЬКО на заданный вопрос. Не добавляй информацию, которой нет в контексте.
- ❗ Проверяй грамматику и согласование слов.
- ❗ Не дублируй пункты. Если два пункта передают одно и то же — объедини или оставь один.
- ❗ Не используй внутренние индексы типа "п. 2.2.3". Вместо этого пиши: «в разделе „Использование РИД“ формы годового отчета».
- Используй маркированные списки, заголовки 3-го уровня (`###`) и подзаголовки для структурирования.
- Перед финальным ответом проверь, не упустил ли ты какие-либо из следующих аспектов, если они упомянуты в контексте:
  * Даты, сроки, числовые значения.
  * Условия применения правил/льгот/санкций.
  * Перечни (например, список документов, видов деятельности).
  * Исключения из правил.
  * Ссылки на конкретные статьи/пункты нормативных актов.

### Правила по типам вопросов:
1.  **Если вопрос касается сроков**:
    *   Укажи точные даты или периоды, если это возможно по контексту.
    *   Если точные сроки не указаны в контексте, укажи, что информация отсутствует.
2.  **Если вопрос о санкциях или последствиях**:
    *   Укажи ВСЕ последствия при нарушении условий и при каких условиях они наступают.
    *   Если последствия не указаны в контексте, сообщи об этом.
3.  **Если вопрос о документах**:
    *   Перечисли ВСЕ необходимые документы и условия их подачи.
    *   Если документы не упомянуты в контексте, сообщи, что информация отсутствует.
4.  **Если вопрос о льготах**:
    *   Укажи льготу, ее суть, ограничения, условия начала и утраты.
    *   Если льгота не упомянута в контексте, сообщи, что информация отсутствует.
5.  **Если вопрос о проверках**:
    *   Опиши, как осуществляется проверка, какие критерии и сроки применяются.
    *   Если методика проверки не указана, сообщи об этом.
6.  **Если вопрос о деятельности**:
    *   Укажи, что разрешено, а что запрещено, включая все упомянутые ограничения.
    *   Если условия деятельности не упомянуты в контексте, сообщи об этом.
7.  **Указывай нормативные акты**:
    *   Всегда включай ссылки на ВСЕ **нормативные акты**, упомянутые в контексте, с полным названием, номером, датой и пунктом (например: «Приказ Фонда «Сколково» от 26.12.2024 № 337-Пр, п. 4 ст. 3»).
    *   ❗ Не пиши просто «Порядок» или «Правила» — это недопустимо.
8.  **Формулы:**
    *   Если для пояснения ответа требуются математические формулы, найди их текстовое представление в контексте и воспроизведи его дословно.
    *   Предпочтительно использовать формат LaTeX, заключая формулы в двойные знаки доллара `$$...$$` для отдельных строк или одинарные `$...$` для формул в тексте, если это соответствует оригиналу в контексте.
    *   Не пытайся сам переписывать или упрощать формулы. Выведи их точно так, как они указаны в предоставленном контексте.
    *   Пример: Если в контексте встречается `Формула 1: Степень = (A / B) * 100%`, выведи именно `$$\text{{Степень}} = \left( \frac{{A}}{{B}} \right) \times 100\%$$` (если это LaTeX в контексте) или `Формула 1: Степень = (A / B) * 100%` (если это текстовое описание).

### Примеры:

---
Пример вопроса: До какого числа нужно подать годовой отчет за 2024 год в Фонд «Сколково»?
Пример ответа:
Годовой отчет за 2024 год должен быть подан не позднее **7 апреля 2025 года**.

**Нормативные акты:**
* Приказ Фонда «Сколково» от 26.12.2024 № 337-Пр, п. 4 ст. 3.

---
Пример вопроса: Что будет, если опоздать со сдачей годового отчета?
Пример ответа:
### Последствия несвоевременной сдачи годового отчета

1.  **Утрата права на грантовую поддержку:**
    *   **Условие:** Представление отчета после 23 ч. 59 мин. по московскому времени 7 апреля 2025 года, но в соответствии со статьей 4 Порядка.
    *   **Последствие:** Утрата права на получение грантовой поддержки Фонда (грантов и микрогрантов) в течение 12 месяцев с даты установления нарушения, предшествовавших подаче заявки на предоставление гранта, минигранта или микрогранта.
    *   **Нормативные акты:** Подпункт «а» пункта 9 статьи 2 Порядка, подпункт 15 пункта 3 статьи 2 Положения о грантах участникам проекта «Сколково», подпункт 11 пункта 1 статьи 3 Положения о микрогрантах участникам проекта «Сколково».

2.  **Досрочное лишение статуса участника проекта:**
    *   **Условие:** Непредставление отчета (включая представление отчета за пределами установленных сроков, за исключением случая, предусмотренного подпунктом «а» настоящего пункта), а равно представление участником проекта заведомо ложных информации или данных в отчете и (или) документах, приложенных к отчету.
    *   **Последствие:** Досрочное лишение юридического лица статуса участника проекта.
    *   **Нормативный акт:** Подпункт «б» пункта 9 статьи 2 Порядка.

3.  **Решение о досрочном исключении из реестра участников проекта:**
    *   **Условие:** Предоставление участником проекта ошибочных или неполных данных или информации в отчете и (или) документах, приложенных к отчету.
    *   **Последствие:** Фонд вправе принять решение о досрочном исключении юридического лица из реестра участников проекта.
    *   **Нормативный акт:** Подпункт «в» пункта 9 статьи 2 Порядка.

**Примечание:** Если участник проекта не направил отчет в срок, предусмотренный пунктом 4 статьи 3 Порядка, либо направленный отчет не был принят Фондом, на адрес электронной почты участника проекта не позднее 28 апреля 2025 года направляется предписание с требованием представить отчет в установленный в предписании срок. После направления предписания участнику проекта на срок, указанный в предписании, открывается форма отчета в личном кабинете.

**Нормативные акты:**
* Приказ Фонда «Сколково» от 26.12.2024 № 337-Пр, ст. 2, ст. 3.
* Положение о грантах участникам проекта «Сколково».
* Положение о микрогрантах участникам проекта «Сколково».

---
Контекст:
{context}

Вопрос:
{question}

Ответ (официально, ясно, по делу):
"""),
    ("human", "{question}"),
])
            # Создание цепочки
            rag_chain = (
                RunnableLambda(retrieve_and_format_sources) # Получаем контекст и источники
                | {
                    "context": lambda x: x["context"],
                    "question": lambda x: x["question"],
                    "source_info": lambda x: x["source_info"],
                    "source_documents": lambda x: x["source_documents"],
                }
                | improved_prompt
                | llm
                | StrOutputParser()
            )

            # --- ШАГ 5: Получение и обработка ответа ---
            result = rag_chain.invoke({"question": question})
            
            # Извлекаем информацию из промежуточного шага
            # ... (внутри метода query_rag, после получения ответа от LLM)
# Пост-обработка
# ...
            # Это немного хак, но работает в рамках текущей структуры
            intermediate_data = retrieve_and_format_sources({"question": question})
            source_info = intermediate_data["source_info"]
            source_documents = intermediate_data["source_documents"]

            answer = result.strip()

            answer = self.post_process_answer(answer)


            # Проверка на бесполезный ответ (упрощена)
            def is_answer_useful(answer_text: str) -> bool:
                return not ("информация не найдена" in answer_text.lower() or "не указано в контексте" in answer_text.lower())

            if not is_answer_useful(answer):
                answer = "К сожалению, точной информации не найдено в доступных нормативных документах."

            response = {
                "answer": answer,
                "sources": source_info,
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

# --- ОСТАЛЬНЫЕ КЛАССЫ БЕЗ ИЗМЕНЕНИЙ ---
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
    
        # Добавьте этот метод в class UserManager:
    @staticmethod
    def delete_user(username_to_delete):
        """Удаляет пользователя по имени."""
        try:
            users_data = UserManager.load_users()
            original_count = len(users_data.get("users", []))
            
            # Фильтруем список пользователей, исключая удаляемого
            users_data["users"] = [user for user in users_data.get("users", []) if user["username"] != username_to_delete]
            
            new_count = len(users_data.get("users", []))
            
            # Если пользователь был найден и удален
            if new_count < original_count:
                if UserManager.save_users(users_data):
                    return True, f"Пользователь {username_to_delete} успешно удален."
                else:
                    return False, "Ошибка при сохранении изменений."
            else:
                return False, f"Пользователь {username_to_delete} не найден."
        except Exception as e:
            return False, f"Ошибка при удалении пользователя: {e}"

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

                # --- ВКЛАДКА "Пользователи" ---
        with admin_tabs[0]:
            st.subheader("👥 Управление пользователями")
            users_data = self.user_manager.load_users()
            users_list = users_data.get("users", [])
            
            if users_list:
                # Отображение пользователей в таблице
                for i, user in enumerate(users_list):
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1]) # Добавлена колонка для кнопки удаления
                    with col1:
                        st.write(f"**{user['username']}**")
                    with col2:
                        st.write(user['role'])
                    with col3:
                        st.write("● Активен") # Предполагается, что все загруженные пользователи активны
                    with col4:
                        # Показываем роль текущего пользователя для контекста
                        if user["username"] == st.session_state.get("username"):
                            st.caption("(Вы)")
                    with col5:
                        # Кнопка удаления
                        # Нельзя удалить самого себя или администратора
                        current_user = st.session_state.get("username")
                        if user["username"] == current_user:
                            st.button("❌", key=f"delete_self_{user['username']}", disabled=True, help="Нельзя удалить самого себя")
                        elif user["role"] == "admin":
                            st.button("🛡️", key=f"delete_admin_{user['username']}", disabled=True, help="Нельзя удалить администратора")
                        else:
                            # Используем st.form для безопасного удаления
                            delete_key = f"delete_form_{user['username']}"
                            # Streamlit не позволяет вложить кнопки в st.form внутри columns напрямую очень удобно.
                            # Поэтому используем st.form_submit_button в отдельном контексте.
                            # Альтернатива: использовать session_state и callback.
                            # Реализуем через session_state и callback для простоты.
                            
                            # Проверим, не был ли запрос на удаление этого пользователя уже сделан в этой сессии
                            delete_confirm_key = f"confirm_delete_{user['username']}"
                            if st.button("🗑️", key=f"delete_btn_{user['username']}", help=f"Удалить пользователя {user['username']}"):
                                # Показываем подтверждение
                                st.session_state[delete_confirm_key] = True
                            
                            if st.session_state.get(delete_confirm_key):
                                st.markdown(
                                    f"""
                                    <div style="background-color: #343A40; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">
                                        <p style="margin: 0; font-size: 16px;">Вы уверены, что хотите удалить пользователя <strong>{user['username']}</strong>?</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )                                
                                col_yes, col_no = st.columns(2)
                                with col_yes:
                                    if st.button("✅", key=f"confirm_yes_{user['username']}", help="Подтвердить удаление"):
                                        # Выполняем удаление
                                        success, message = self.user_manager.delete_user(user['username'])
                                        if success:
                                            st.success(message)
                                            # Очищаем флаг подтверждения
                                            if delete_confirm_key in st.session_state:
                                                del st.session_state[delete_confirm_key]
                                            # Перезагружаем страницу, чтобы обновить список
                                            st.rerun()
                                        else:
                                            st.error(message)
                                            # Очищаем флаг подтверждения
                                            if delete_confirm_key in st.session_state:
                                                del st.session_state[delete_confirm_key]
                                with col_no:
                                    if st.button("❌", key=f"confirm_no_{user['username']}", help="Отменить удаление"): # Добавляем всплывающую подсказку
                                        # Просто убираем предупреждение
                                        if delete_confirm_key in st.session_state:
                                            del st.session_state[delete_confirm_key]
                                        st.rerun() # Перерисовываем, чтобы убрать кнопки подтверждения

            else:
                st.info("Пользователи не найдены.")

            st.divider()
            
            # --- Форма добавления нового пользователя ---
            with st.form("add_user_form"):
                new_username = st.text_input("Логин")
                new_password = st.text_input("Пароль", type="password")
                new_role = st.selectbox("Роль", ["user", "editor", "admin"])
                submit_button = st.form_submit_button("Добавить пользователя")
                if submit_button:
                    if new_username and new_password:
                        # users_data = self.user_manager.load_users() # Уже загружено выше
                        if any(user["username"] == new_username for user in users_list):
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
                                st.rerun() # Перезагружаем для отображения нового пользователя
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
            st.set_page_config(page_title=CONFIG["app"]["name"], page_icon=CONFIG["app"]["page_icon"], layout="centered")
        else:
            st.set_page_config(page_title=CONFIG["app"]["name"], page_icon=CONFIG["app"]["page_icon"], layout="wide")

        if st.session_state.role == "admin":
            page = st.sidebar.selectbox("Навигация", ["Чат", "Админ-панель"])
        else:
            page = "Чат"

        if st.session_state.role == "user":
            st.title(f"🏢 {CONFIG['app']['name']}")
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
                st.title(f"🏢 {CONFIG['app']['name']}")
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

        if "indexed_on_startup" not in st.session_state:
            with st.spinner("Проверяем индексацию документов..."):
                self.rag_system.document_processor.auto_index_all_pdfs()
            st.session_state.indexed_on_startup = True

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
                if os.path.exists(CONFIG["chroma"]["persist_directory"]):
                    st.success("✅ База знаний загружена")
                    indexed_files = self.rag_system.document_processor.get_indexed_files()
                    if indexed_files:
                        st.subheader("Индексированные документы:")
                        for filename in indexed_files:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📄 {filename}")
                            with col2:
                                if st.session_state.role == "admin":
                                    if st.button("❌", key=f"delete_{filename}", help=f"Удалить {filename}"):
                                        if self.rag_system.document_processor.remove_document_from_index(filename):
                                            st.rerun()
                                else:
                                    st.write("")
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
                            self.rag_system.document_processor.index_pdf(file_path)
                            st.rerun()

                if st.session_state.role == "admin":
                    st.divider()
                    if st.button("🔄 Обновить индекс"):
                        with st.spinner("Обновляем индекс..."):
                            self.rag_system.document_processor.auto_index_all_pdfs()
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

if __name__ == "__main__":
    app = SkolkovoConsultantApp()
    app.main()
