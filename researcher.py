# researcher.py (v1.2.0 - Migrated to google-genai)
import asyncio
import json
import logging
from typing import List, Dict, Any, Callable, Coroutine, Optional

# --- ИЗМЕНЕНИЯ ИМПОРТОВ ---
from google import genai
from google.genai import types as genai_types # Используем типы нового SDK
from google.api_core import exceptions as google_api_exceptions # Для обработки ошибок API
from pydantic import BaseModel, Field, ValidationError

try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

logger = logging.getLogger(__name__)

# --- Модели Pydantic (без изменений) ---
class SearchQuery(BaseModel):
    sub_query: str = Field(description="Конкретный подзапрос для поиска информации по аспекту основной темы.")
    purpose: str = Field(description="Краткое объяснение, какую информацию должен найти этот подзапрос.")

class ResearchPlan(BaseModel):
    query_plan: List[SearchQuery] = Field(description="Список поисковых подзапросов для исследования темы.")

class ExtractedKnowledge(BaseModel):
    key_insights: List[str] = Field(description="Список ключевых фактов, данных или выводов, извлеченных из текста.")
    source_summary: str = Field(description="Очень краткое резюме (1-2 предложения) основной сути источника.")

class ReflectionResult(BaseModel):
    information_gaps: List[str] = Field(description="Список конкретных пробелов в информации или областей, требующих дальнейшего изучения.")
    new_queries: List[SearchQuery] = Field(description="Список новых поисковых подзапросов для заполнения пробелов.")
    is_complete: bool = Field(description="Флаг, указывающий, достаточно ли информации для ответа на исходный запрос.")

# --- Класс ошибки (без изменений) ---
class ResearchError(Exception):
    pass

# --- Основной класс (значительно переработан) ---
class DeepResearcher:
    def __init__(
        self,
        # api_key: str, # <-- УБРАНО: Ключ берется из окружения клиентом
        model_name: str, # Имя модели для использования в запросах
        depth: int = 2,
        breadth: int = 3,
        max_completion_tokens: int = 8192, # Оставляем как параметр класса
        temperature: float = 0.3, # Оставляем как параметр класса
        request_timeout: int = 300 # Таймаут для запросов к API
    ):
        self.depth = depth
        self.breadth = breadth
        # Сохраняем базовое имя модели (без префикса 'models/')
        self.model_name = model_name.replace("models/", "")
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        # self.api_key = api_key # <-- УБРАНО

        # --- Инициализация клиента ---
        try:
            # Клиент ищет ключ GOOGLE_API_KEY в окружении
            self.client = genai.Client()
            # Проверочный вызов для валидации ключа и соединения
            self.client.models.list(page_size=1) # Достаточно запросить одну модель
            logger.info("Клиент Google GenAI SDK успешно инициализирован.")
        except google_api_exceptions.PermissionDenied as e:
             logger.exception(f"Ошибка прав доступа (проверьте API ключ): {e}")
             raise ResearchError(f"Не удалось настроить Google GenAI SDK: Неверный API ключ или нет прав доступа. {e}") from e
        except Exception as e:
             logger.exception(f"Ошибка инициализации клиента Google GenAI SDK: {e}")
             raise ResearchError(f"Не удалось настроить Google GenAI SDK: {e}") from e

        # --- УБРАНО: Конфигурации и модели создаются при вызове ---
        # self.default_generation_config = ...
        # self.json_generation_config = ...
        # self.search_tool = ...
        # self.model = ...
        # self.report_model = ...

        self.all_knowledge: List[Dict[str, Any]] = []
        self.queries_history: List[str] = []
        self.log_callback: Callable[[str], Any] | None = None

    async def _log(self, message: str):
        """Логирует сообщение и вызывает callback, если он есть."""
        logger.info(message)
        if self.log_callback:
            try:
                # Проверяем, является ли callback корутиной
                if asyncio.iscoroutinefunction(self.log_callback):
                    # Запускаем корутину как задачу, чтобы не блокировать основной поток
                    asyncio.create_task(self.log_callback(message))
                else:
                    # Вызываем синхронный callback напрямую
                    self.log_callback(message)
            except Exception as e:
                # Логируем ошибку в самом callback, но не прерываем основной процесс
                logger.error(f"Ошибка при вызове log_callback: {e}")

    async def _call_gemini(
        self,
        prompt: str,
        response_schema: Optional[type[BaseModel]] = None,
        use_search_tool: bool = False,
        # model_name передается извне или используется self.model_name
    ) -> str | Dict[str, Any]:
        """
        Выполняет вызов Google GenAI API с обработкой ошибок и парсингом JSON.
        Использует self.model_name для определения модели.
        """
        # Формируем полное имя модели с префиксом
        full_model_name = f'models/{self.model_name}'

        config_dict = {
            'temperature': self.temperature,
            'max_output_tokens': self.max_completion_tokens
        }
        tools_list = []

        if use_search_tool:
            # ВАЖНО: Проверить совместимость поиска и JSON в конкретной модели, если нужно
            if response_schema:
                await self._log(f"Предупреждение: Поиск GoogleSearch не может использоваться одновременно с запросом JSON схемы для модели {self.model_name}. Поиск отключен.")
                # use_search_tool = False # Не меняем флаг, просто не добавляем инструмент
            else:
                # Используем GoogleSearch() для моделей Gemini 2.0+
                # Для старых (1.5) может потребоваться GoogleSearchRetrieval, но SDK должен сам разобраться
                tools_list.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
                await self._log("Инструмент поиска GoogleSearch активирован.")

        if response_schema:
            config_dict['response_mime_type'] = 'application/json'
            # Передаем сам класс Pydantic модели как схему
            config_dict['response_schema'] = response_schema
            await self._log(f"Запрошен JSON ответ со схемой: {response_schema.__name__}")
            # Не добавляем описание схемы в промпт, SDK делает это сам

        if tools_list:
             config_dict['tools'] = tools_list

        # Создаем объект конфигурации
        try:
            generation_config = genai_types.GenerateContentConfig(**config_dict)
        except Exception as e:
             await self._log(f"Ошибка создания GenerateContentConfig: {e}")
             raise ResearchError(f"Ошибка конфигурации запроса к Gemini: {e}") from e

        try:
            await self._log(f"Отправка запроса к GenAI (Модель: {full_model_name}, Схема: {response_schema is not None}, Поиск: {use_search_tool})...")

            # Используем асинхронный клиент
            response = await self.client.aio.models.generate_content(
                model=full_model_name,
                contents=prompt, # Передаем промпт как строку
                generation_config=generation_config,
                request_options={'timeout': self.request_timeout}
            )

            # --- Обработка ответа ---
            # 1. Проверка на блокировку и пустой ответ
            if not response.candidates:
                 safety_feedback = getattr(response, 'prompt_feedback', None)
                 block_reason = getattr(safety_feedback, 'block_reason', 'Неизвестно')
                 await self._log(f"GenAI вернул пустой ответ. Причина блокировки: {block_reason}. Safety Feedback: {safety_feedback}")
                 raise ResearchError(f"Запрос к GenAI заблокирован (Причина: {block_reason}) или вернул пустой ответ.")

            # 2. Обработка JSON ответа (если запрашивался)
            if response_schema:
                response_text = response.text # Получаем текст ответа
                await self._log(f"Получен текстовый ответ для JSON схемы (длина: {len(response_text)}). Попытка парсинга...")
                # Убираем возможные ```json маркеры
                cleaned_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
                if not cleaned_text:
                     raise ResearchError("GenAI вернул пустой текст при запросе JSON.")

                try:
                    # Пытаемся распарсить стандартными средствами
                    parsed_response = response_schema.model_validate_json(cleaned_text)
                    await self._log("Ответ GenAI успешно распарсен в JSON.")
                    return parsed_response.model_dump()
                except (ValidationError, json.JSONDecodeError) as e:
                    await self._log(f"Ошибка парсинга JSON от GenAI: {e}. Ответ: {cleaned_text[:500]}...")
                    # Пытаемся исправить с помощью json_repair, если он доступен
                    if HAS_JSON_REPAIR:
                        try:
                            await self._log("Попытка исправить JSON с помощью json_repair...")
                            repaired_json_str = json_repair.repair_json(cleaned_text)
                            parsed_response = response_schema.model_validate_json(repaired_json_str)
                            await self._log("JSON успешно исправлен и распарсен.")
                            return parsed_response.model_dump()
                        except Exception as repair_error:
                             await self._log(f"Не удалось исправить JSON: {repair_error}")
                             # Поднимаем исходную ошибку парсинга
                             raise ResearchError(f"Не удалось распарсить или исправить JSON ответ от GenAI: {e}") from e
                    else:
                         # Если json_repair не установлен, просто поднимаем ошибку
                         raise ResearchError(f"Не удалось распарсить JSON ответ от GenAI (json_repair не найден): {e}") from e

            # 3. Обработка обычного текстового ответа (или ответа с поиском)
            else:
                # Новый SDK может включать результаты поиска прямо в текст
                # Проверяем, есть ли текст
                if response.text:
                    await self._log(f"Получен текстовый ответ от GenAI (длина: {len(response.text)}).")
                    return response.text
                else:
                    # Если текста нет, но и не JSON, возможно, был только вызов инструмента?
                    # Проверим наличие function_calls (хотя для google_search это редкость)
                    function_calls = getattr(response.candidates[0].content, 'parts', [])
                    fc_parts = [p for p in function_calls if hasattr(p, 'function_call')]
                    if fc_parts:
                         await self._log("GenAI вернул вызов функции, но нет текстового ответа.")
                         # Вернем пустую строку или специальное сообщение
                         return "" # Или "Выполнен поиск, но нет текстового резюме."
                    else:
                         await self._log("GenAI вернул ответ без текста и без JSON.")
                         raise ResearchError("GenAI вернул пустой текстовый ответ.")

        # --- Обработка ошибок API ---
        except google_api_exceptions.PermissionDenied as e:
            await self._log(f"Ошибка прав доступа при вызове GenAI API: {e}")
            raise ResearchError(f"Ошибка API ключа или прав доступа: {e.message}") from e
        except google_api_exceptions.ResourceExhausted as e:
            await self._log(f"Превышена квота GenAI API: {e}")
            raise ResearchError(f"Превышена квота Google API: {e.message}") from e
        except google_api_exceptions.InvalidArgument as e:
             await self._log(f"Неверный аргумент при вызове GenAI API: {e}")
             # Проверяем на частые проблемы
             if "API key not valid" in str(e):
                  raise ResearchError("Ошибка API ключа Gemini. Проверьте ключ.") from e
             if "JSON mode is not supported" in str(e) or "Tool use is not supported" in str(e):
                  raise ResearchError(f"Выбранная модель '{self.model_name}' не поддерживает JSON mode или инструменты (поиск).") from e
             raise ResearchError(f"Неверный аргумент запроса к Google API: {e.message}") from e
        except google_api_exceptions.GoogleAPIError as e:
            await self._log(f"Общая ошибка Google API: {e}")
            raise ResearchError(f"Ошибка при взаимодействии с Google API: {e.message}") from e
        except Exception as e:
            await self._log(f"Непредвиденная ошибка при вызове GenAI API: {e}")
            logger.exception("Детали непредвиденной ошибки GenAI API:") # Логируем полный traceback
            raise ResearchError(f"Непредвиденная ошибка при взаимодействии с GenAI API: {e}") from e

    # --- Методы логики исследования (обновлены вызовы _call_gemini) ---

    async def _generate_initial_queries(self, topic: str) -> List[SearchQuery]:
        """Генерирует начальные поисковые запросы."""
        prompt = f"""
        Вы - ИИ-ассистент для планирования исследований.
        Ваша задача - разбить основную тему исследования "{topic}" на {self.breadth} конкретных, целенаправленных поисковых подзапросов.
        Каждый подзапрос должен исследовать отдельный аспект темы и иметь четкую цель.
        Избегайте слишком широких или общих запросов.
        Предоставьте результат в формате JSON согласно схеме ResearchPlan.
        """
        await self._log(f"Генерация начальных запросов для темы: {topic}")
        # Запрашиваем JSON, поиск не нужен
        response_data = await self._call_gemini(
            prompt,
            response_schema=ResearchPlan,
            use_search_tool=False
        )
        # Проверка типа остается, так как _call_gemini возвращает dict для JSON
        if isinstance(response_data, dict) and "query_plan" in response_data:
             try:
                 plan = ResearchPlan.model_validate(response_data)
                 self.queries_history.extend([q.sub_query for q in plan.query_plan])
                 await self._log(f"Сгенерировано {len(plan.query_plan)} начальных запросов.")
                 return plan.query_plan
             except ValidationError as e:
                  await self._log(f"Ошибка валидации Pydantic для ResearchPlan: {e}. Данные: {response_data}")
                  raise ResearchError("Не удалось валидировать план запросов от Gemini.") from e
        else:
            await self._log(f"Не удалось сгенерировать начальный план запросов. Получен неожиданный ответ: {type(response_data)}")
            raise ResearchError("Не удалось сгенерировать начальный план запросов (неверный формат ответа).")

    async def _search_and_extract(self, query: SearchQuery) -> Dict[str, Any]:
        """
        Выполняет поиск по подзапросу (Шаг 1) и извлекает знания (Шаг 2).
        """
        # --- Шаг 1: Поиск информации (используем инструмент поиска) ---
        search_prompt = f"""
        Используя поиск Google (инструмент google_search), найди и кратко изложи наиболее релевантную информацию по следующему запросу:
        Запрос: "{query.sub_query}"
        Цель поиска: "{query.purpose}"
        Предоставь результаты в виде связного текста или списка ключевых находок.
        """
        await self._log(f"Шаг 1: Выполнение поиска для: '{query.sub_query}' (Цель: {query.purpose})")
        # Запрашиваем текст, используем поиск
        search_results_text = await self._call_gemini(
            search_prompt,
            response_schema=None,
            use_search_tool=True
        )

        # Проверяем, что результат - строка и она не пустая
        if not isinstance(search_results_text, str) or not search_results_text.strip():
            await self._log(f"Поиск не вернул текстовых результатов для запроса: '{query.sub_query}'.")
            # Возвращаем пустой результат, но с исходным запросом
            return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": "Поиск не дал результатов или не вернул текст."}

        await self._log(f"Шаг 1: Получены результаты поиска (длина: {len(search_results_text)}).")

        # --- Шаг 2: Извлечение знаний из результатов поиска (запрос JSON) ---
        extract_prompt = f"""
        Вы - ИИ-ассистент для извлечения информации.
        Проанализируйте предоставленный ниже текст (результаты поиска Google по запросу "{query.sub_query}" с целью "{query.purpose}") и извлеките ключевые факты, данные и выводы, относящиеся к цели запроса.
        Предоставьте результат в формате JSON согласно схеме ExtractedKnowledge.

        Текст для анализа:
        ```
        {search_results_text}
        ```
        """
        await self._log(f"Шаг 2: Извлечение знаний из результатов поиска для: '{query.sub_query}'")
        # Запрашиваем JSON, поиск здесь не нужен
        response_data = await self._call_gemini(
            extract_prompt,
            response_schema=ExtractedKnowledge,
            use_search_tool=False
        )

        if isinstance(response_data, dict):
             try:
                 knowledge = ExtractedKnowledge.model_validate(response_data)
                 await self._log(f"Шаг 2: Извлечено знаний: {len(knowledge.key_insights)} ключевых моментов.")
                 knowledge_dict = knowledge.model_dump()
                 knowledge_dict["original_query"] = query.sub_query
                 knowledge_dict["purpose"] = query.purpose
                 return knowledge_dict
             except ValidationError as e:
                  await self._log(f"Ошибка валидации Pydantic для ExtractedKnowledge: {e}. Данные: {response_data}")
                  # Возвращаем пустой результат, но с исходным запросом
                  return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": f"Не удалось валидировать извлеченные знания: {e}"}
        else:
            await self._log(f"Шаг 2: Не удалось извлечь знания для запроса: '{query.sub_query}'. Получен неожиданный ответ: {type(response_data)}")
            return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": "Не удалось извлечь информацию (неверный формат ответа)."}

    async def _reflect_and_plan_next(self, topic: str) -> ReflectionResult:
        """Анализирует собранные знания и планирует следующие шаги."""
        # Собираем резюме знаний
        knowledge_summary = "\n".join(
            f"- По запросу '{k.get('original_query', 'N/A')}': {k.get('source_summary', 'N/A')}. Ключевые моменты: {'; '.join(k.get('key_insights', []))}"
            for k in self.all_knowledge if k.get('key_insights') # Учитываем только те, где есть инсайты
        )
        history_str = ", ".join(self.queries_history)

        prompt = f"""
        Вы - ИИ-ассистент для рефлексии и планирования исследований.
        Ваша задача - проанализировать собранную информацию по теме "{topic}", определить пробелы и спланировать следующие шаги исследования.

        Собранные знания:
        {knowledge_summary if knowledge_summary else "Пока нет собранных знаний."}

        История выполненных запросов: {history_str}

        Проанализируйте текущие знания:
        1. Достаточно ли информации для полного ответа на исходный запрос "{topic}"?
        2. Какие ключевые аспекты темы еще не раскрыты или раскрыты недостаточно?
        3. Какие новые вопросы возникли на основе собранной информации?

        На основе анализа сгенерируйте результат в формате JSON согласно схеме ReflectionResult.
        - Если информации достаточно, установите is_complete = true и оставьте new_queries пустым.
        - Если информации недостаточно, установите is_complete = false, опишите пробелы в information_gaps и предложите до {self.breadth} новых, конкретных подзапросов в new_queries для их заполнения. Новые запросы не должны повторять запросы из истории.
        """
        await self._log("Рефлексия над собранными знаниями и планирование следующих шагов...")
        # Запрашиваем JSON, поиск не нужен
        response_data = await self._call_gemini(
            prompt,
            response_schema=ReflectionResult,
            use_search_tool=False
        )

        if isinstance(response_data, dict):
            try:
                reflection = ReflectionResult.model_validate(response_data)
                await self._log(f"Результат рефлексии: Завершено={reflection.is_complete}, Новых запросов={len(reflection.new_queries)}")
                return reflection
            except ValidationError as e:
                 await self._log(f"Ошибка валидации Pydantic для ReflectionResult: {e}. Данные: {response_data}")
                 raise ResearchError("Не удалось валидировать результат рефлексии от Gemini.") from e
        else:
             await self._log(f"Не удалось получить результат рефлексии. Получен неожиданный ответ: {type(response_data)}")
             raise ResearchError("Не удалось получить результат рефлексии (неверный формат ответа).")

    async def _generate_final_report(self, topic: str) -> str:
        """Генерирует финальный отчет на основе всех собранных знаний."""
        # Собираем детализированные знания
        knowledge_details = ""
        valid_knowledge_count = 0
        for i, k in enumerate(self.all_knowledge):
            # Пропускаем записи без ключевых моментов
            if not k.get('key_insights'):
                continue
            valid_knowledge_count += 1
            insights = "\n  - ".join(k.get('key_insights', []))
            knowledge_details += (
                f"\nИсточник {valid_knowledge_count} (Результат поиска по запросу: '{k.get('original_query', 'N/A')}')\n"
                f"Резюме источника: {k.get('source_summary', 'N/A')}\n"
                f"Ключевые моменты:\n  - {insights}\n" # Убрали проверку на пустые инсайты здесь, т.к. отфильтровали выше
            )

        if not knowledge_details:
             await self._log("Нет собранных знаний для генерации отчета.")
             return "Не удалось собрать достаточно информации для генерации отчета."

        prompt = f"""
        Вы - ИИ-аналитик и писатель отчетов.
        Ваша задача - написать исчерпывающий, структурированный и хорошо аргументированный отчет в формате Markdown по теме "{topic}", основываясь ИСКЛЮЧИТЕЛЬНО на предоставленных ниже знаниях, извлеченных из веб-поиска.

        Собранные знания из различных источников ({valid_knowledge_count} шт.):
        {knowledge_details}

        Требования к отчету:
        - Структура: Введение (контекст и цель), Основная часть (логически сгруппированные подразделы с анализом информации из источников), Заключение (основные выводы, синтез информации).
        - Содержание: Глубоко проанализируйте и синтезируйте информацию из разных источников. Сравнивайте точки зрения, если они различаются. Выделяйте ключевые выводы и закономерности. Не просто перечисляйте факты, а объясняйте их значение и связь.
        - Язык: Объективный, аналитический, ясный и точный.
        - Цитирование: НЕ используйте номера источников или прямые ссылки. Вся информация уже считается полученной из предоставленных знаний.
        - Полнота: Используйте ВСЕ предоставленные знания.
        - Формат: Чистый Markdown. Используйте заголовки (#, ##, ###), списки (* или -), выделение текста (*курсив*, **жирный**).

        Напишите финальный отчет.
        """
        await self._log("Генерация финального отчета...")
        # Запрашиваем текст, поиск не нужен
        report = await self._call_gemini(
            prompt,
            response_schema=None,
            use_search_tool=False
        )

        if isinstance(report, str):
            await self._log("Финальный отчет сгенерирован.")
            # Убираем возможные ```markdown маркеры
            cleaned_report = report.strip().removeprefix("```markdown").removesuffix("```").strip()
            return cleaned_report
        else:
            await self._log(f"Ошибка: генерация отчета вернула не строку, а {type(report)}.")
            raise ResearchError("Ошибка при генерации финального отчета (неверный тип ответа).")

    async def research(self, topic: str, log_callback: Optional[Callable[[str], Any]] = None) -> str:
        """
        Основной метод для запуска итеративного исследования.
        """
        self.log_callback = log_callback
        self.all_knowledge = []
        self.queries_history = []

        try:
            await self._log(f"--- Начало исследования по теме: '{topic}' (Модель: {self.model_name}) ---")
            current_queries = await self._generate_initial_queries(topic)

            if not current_queries:
                 await self._log("Не удалось сгенерировать начальные запросы. Исследование прервано.")
                 return "Ошибка: Не удалось спланировать начальные шаги исследования."

            for i in range(self.depth):
                iteration_num = i + 1
                await self._log(f"--- Итерация исследования {iteration_num}/{self.depth} ---")
                if not current_queries:
                     await self._log(f"Нет запросов для выполнения на итерации {iteration_num}. Завершение.")
                     break

                await self._log(f"Запросы для итерации {iteration_num}: {[q.sub_query for q in current_queries]}")

                # Запускаем задачи поиска и извлечения параллельно
                search_tasks = [self._search_and_extract(query) for query in current_queries]
                # Используем gather для ожидания всех задач
                knowledge_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                new_knowledge_count = 0
                successful_results = 0
                iteration_errors = []
                for result in knowledge_results:
                    if isinstance(result, dict):
                        successful_results += 1
                        # Добавляем результат, даже если key_insights пуст, но есть summary
                        self.all_knowledge.append(result)
                        if result.get("key_insights"):
                            new_knowledge_count += len(result["key_insights"])
                        else:
                             await self._log(f"Нет ключевых моментов для запроса: '{result.get('original_query')}' (Резюме: {result.get('source_summary', 'N/A')[:50]}...)")
                    elif isinstance(result, Exception):
                         # Логируем ошибку, но продолжаем итерацию
                         error_msg = f"Ошибка при поиске/извлечении на итерации {iteration_num}: {result}"
                         await self._log(error_msg)
                         iteration_errors.append(error_msg)
                         # Можно добавить ошибку в общий лог задачи, если нужно
                         # if self.log_callback: asyncio.create_task(self.log_callback(error_msg))
                    else:
                         # Неожиданный тип результата
                         await self._log(f"Неожиданный результат поиска/извлечения: {type(result)}")

                await self._log(f"Итерация {iteration_num}: Успешно обработано {successful_results}/{len(current_queries)} запросов. Добавлено {new_knowledge_count} новых ключевых моментов.")
                if iteration_errors:
                     await self._log(f"Итерация {iteration_num}: Обнаружены ошибки ({len(iteration_errors)} шт.).")

                # Проверяем, есть ли смысл продолжать, если первая итерация провалилась
                if not self.all_knowledge and iteration_num == 1:
                    await self._log("Не удалось собрать информацию на первой итерации. Прерывание исследования.")
                    return "Не удалось найти релевантную информацию по вашему запросу на начальном этапе."

                # Рефлексия и планирование следующей итерации (если не последняя)
                if iteration_num < self.depth:
                    try:
                        reflection = await self._reflect_and_plan_next(topic)
                        if reflection.is_complete:
                            await self._log("Исследование завершено досрочно по результатам рефлексии (is_complete=True).")
                            break
                        elif not reflection.new_queries:
                             await self._log("Исследование завершено досрочно по результатам рефлексии (нет новых запросов).")
                             break
                        else:
                            # Фильтруем новые запросы, чтобы избежать повторов
                            new_unique_queries = []
                            for nq in reflection.new_queries:
                                if nq.sub_query not in self.queries_history:
                                    new_unique_queries.append(nq)
                                    self.queries_history.append(nq.sub_query) # Добавляем в историю сразу

                            if not new_unique_queries:
                                 await self._log("Нет новых уникальных запросов для следующей итерации. Завершение.")
                                 break

                            # Ограничиваем количество запросов шириной (breadth)
                            current_queries = new_unique_queries[:self.breadth]
                            await self._log(f"Планирование итерации {iteration_num + 1} с {len(current_queries)} новыми запросами.")

                    except ResearchError as reflect_error:
                         await self._log(f"Ошибка на этапе рефлексии итерации {iteration_num}: {reflect_error}. Продолжение без новых запросов.")
                         # Если рефлексия не удалась, прерываем цикл, так как нет новых запросов
                         break
                else:
                     await self._log(f"Достигнута максимальная глубина исследования ({self.depth}). Переход к генерации отчета.")

            # Генерация финального отчета после всех итераций
            await self._log("--- Генерация финального отчета ---")
            if not self.all_knowledge:
                 await self._log("Не удалось собрать информацию за все итерации.")
                 # Можно вернуть более информативное сообщение
                 return "Не удалось найти релевантную информацию по вашему запросу после выполнения всех шагов исследования."

            final_report = await self._generate_final_report(topic)
            return final_report

        except ResearchError as e:
            # Логируем и возвращаем ошибку исследования
            await self._log(f"Критическая ошибка исследования (ResearchError): {e}")
            # Можно добавить traceback в лог для отладки
            # logger.exception("Traceback ошибки ResearchError:")
            return f"Ошибка в процессе исследования: {e}"
        except Exception as e:
            # Логируем неожиданную ошибку
            logger.exception(f"Непредвиденная критическая ошибка в методе research: {e}")
            await self._log(f"Непредвиденная критическая ошибка: {e}")
            return f"Произошла непредвиденная внутренняя ошибка сервера: {e}"
        finally:
            # Сбрасываем callback после завершения исследования
            self.log_callback = None
            await self._log("--- Исследование завершено (успешно или с ошибкой) ---")
