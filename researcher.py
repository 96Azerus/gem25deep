# researcher.py (v1.1.2)
import asyncio
import json
import logging
from typing import List, Dict, Any, Callable, Coroutine

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ToolConfig # Импортируем нужные типы
from pydantic import BaseModel, Field, ValidationError
try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

logger = logging.getLogger(__name__)

# --- Модели Pydantic ---
# (Схемы SearchQuery, ResearchPlan, ExtractedKnowledge, ReflectionResult остаются без изменений)
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

# --- Класс ошибки ---
class ResearchError(Exception):
    pass

# --- Основной класс ---
class DeepResearcher:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        depth: int = 2,
        breadth: int = 3,
        max_completion_tokens: int = 8192,
        temperature: float = 0.3,
    ):
        self.depth = depth
        self.breadth = breadth
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.api_key = api_key

        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
             logger.exception(f"Ошибка конфигурации Gemini API ключа: {e}")
             raise ResearchError(f"Не удалось настроить Gemini API ключ: {e}") from e

        # Базовая конфигурация генерации
        self.default_generation_config = GenerationConfig(
            max_output_tokens=self.max_completion_tokens,
            temperature=self.temperature,
        )
        # Конфигурация для запроса JSON
        self.json_generation_config = GenerationConfig(
            max_output_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            response_mime_type="application/json"
        )
        # Конфигурация для использования поиска
        self.search_tool = genai.Tool.from_google_search_retrieval(
             google_search_retrieval=genai.protos.GoogleSearchRetrieval(disable_attribution=False)
        )


        try:
            # Инициализируем модель один раз
            self.model = genai.GenerativeModel(
                self.model_name
                # Конфигурацию и инструменты будем передавать в generate_content_async
            )
            logger.info(f"Модель Gemini '{self.model_name}' успешно инициализирована.")
        except Exception as e:
            logger.exception(f"Ошибка инициализации модели Gemini '{self.model_name}': {e}")
            raise ResearchError(f"Не удалось инициализировать модель Gemini '{self.model_name}': {e}") from e

        self.all_knowledge: List[Dict[str, Any]] = []
        self.queries_history: List[str] = []
        self.log_callback: Callable[[str], Any] | None = None

    async def _log(self, message: str):
        logger.info(message)
        if self.log_callback:
            try:
                if asyncio.iscoroutinefunction(self.log_callback):
                    asyncio.create_task(self.log_callback(message))
                else:
                    self.log_callback(message)
            except Exception as e:
                logger.error(f"Ошибка при вызове log_callback: {e}")

    async def _call_gemini(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        use_search_tool: bool = False,
        is_report_generation: bool = False # Оставим флаг, хотя модель одна
    ) -> str | Dict[str, Any]:
        """
        Выполняет вызов Gemini API с обработкой ошибок и парсингом JSON.
        """
        generation_config = self.default_generation_config
        tools_list = None
        final_prompt = prompt

        if response_schema:
            generation_config = self.json_generation_config
            # ИСПРАВЛЕНИЕ: Убран indent=2
            schema_json = json.dumps(response_schema.model_json_schema(), ensure_ascii=False)
            final_prompt = (
                f"{prompt}\n\nПожалуйста, отформатируйте ваш ответ как JSON объект, "
                f"соответствующий следующей Pydantic схеме:\n```json\n{schema_json}\n```"
                "\nВаш ответ должен содержать ТОЛЬКО валидный JSON объект без каких-либо "
                "других текстовых пояснений до или после него."
            )
            # Важно: Не используем поиск при запросе JSON
            if use_search_tool:
                 await self._log("Предупреждение: Поиск отключен, так как запрошен JSON ответ.")
                 use_search_tool = False

        if use_search_tool:
            tools_list = [self.search_tool]

        try:
            await self._log(f"Отправка запроса к Gemini (Модель: {self.model_name}, Схема: {response_schema is not None}, Поиск: {use_search_tool})...")

            response = await self.model.generate_content_async(
                final_prompt,
                tools=tools_list, # Передаем список инструментов
                generation_config=generation_config, # Передаем выбранную конфигурацию
                request_options={'timeout': 300}
            )

            # Обработка ответа... (остается как в v1.1.1, но с исправлением парсинга)
            if not response.candidates or not response.candidates[0].content.parts:
                 await self._log("Gemini вернул пустой ответ.")
                 try:
                     safety_info = response.prompt_feedback
                     await self._log(f"Safety Feedback: {safety_info}")
                 except Exception:
                     pass
                 raise ResearchError("Gemini вернул пустой ответ. Возможно, запрос был заблокирован фильтрами безопасности.")

            response_text = response.text
            await self._log(f"Получен ответ от Gemini (длина: {len(response_text)}).")

            if response_schema:
                cleaned_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
                try:
                    parsed_response = response_schema.model_validate_json(cleaned_text)
                    await self._log("Ответ Gemini успешно распарсен в JSON.")
                    return parsed_response.model_dump()
                except (ValidationError, json.JSONDecodeError) as e:
                    await self._log(f"Ошибка парсинга JSON от Gemini: {e}. Ответ: {cleaned_text[:500]}...")
                    if HAS_JSON_REPAIR:
                        try:
                            repaired_json_str = json_repair.repair_json(cleaned_text)
                            parsed_response = response_schema.model_validate_json(repaired_json_str)
                            await self._log("JSON успешно исправлен и распарсен.")
                            return parsed_response.model_dump()
                        except Exception as repair_error:
                             await self._log(f"Не удалось исправить JSON: {repair_error}")
                             raise ResearchError(f"Не удалось распарсить JSON ответ от Gemini: {e}") from e
                    else:
                         raise ResearchError(f"Не удалось распарсить JSON ответ от Gemini: {e}") from e
            else:
                return response_text

        except Exception as e:
            # Обработка ошибок API... (остается как в v1.1.1)
            await self._log(f"Ошибка при вызове Gemini API: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
                 await self._log(f"Safety Feedback: {e.response.prompt_feedback}")
                 if e.response.prompt_feedback.block_reason:
                     raise ResearchError(f"Запрос к Gemini заблокирован: {e.response.prompt_feedback.block_reason}") from e
            if "API key not valid" in str(e):
                 raise ResearchError("Ошибка API ключа Gemini. Проверьте ключ.") from e
            elif "quota" in str(e).lower():
                 raise ResearchError("Превышена квота Gemini API.") from e
            # Добавим проверку на ошибку, связанную с инструментами/JSON
            elif "doesn't support function calling" in str(e) or "JSON mode is not supported" in str(e):
                 raise ResearchError(f"Выбранная модель '{self.model_name}' не поддерживает одновременное использование инструментов и/или JSON mode.") from e
            raise ResearchError(f"Ошибка при взаимодействии с Gemini API: {e}") from e

    async def _generate_initial_queries(self, topic: str) -> List[SearchQuery]:
        """Генерирует начальные поисковые запросы."""
        prompt = f"""
        Вы - ИИ-ассистент для планирования исследований.
        Ваша задача - разбить основную тему исследования "{topic}" на {self.breadth} конкретных, целенаправленных поисковых подзапросов.
        Каждый подзапрос должен исследовать отдельный аспект темы и иметь четкую цель.
        Избегайте слишком широких или общих запросов.
        """
        await self._log(f"Генерация начальных запросов для темы: {topic}")
        # Запрашиваем JSON, поиск не используем
        response_data = await self._call_gemini(prompt, response_schema=ResearchPlan, use_search_tool=False)
        if isinstance(response_data, dict) and "query_plan" in response_data:
             plan = ResearchPlan.model_validate(response_data)
             self.queries_history.extend([q.sub_query for q in plan.query_plan])
             await self._log(f"Сгенерировано {len(plan.query_plan)} начальных запросов.")
             return plan.query_plan
        else:
            await self._log(f"Не удалось сгенерировать начальный план запросов. Получен ответ: {response_data}")
            raise ResearchError("Не удалось сгенерировать начальный план запросов.")

    async def _search_and_extract(self, query: SearchQuery) -> Dict[str, Any]:
        """
        Выполняет поиск по подзапросу (Шаг 1) и извлекает знания (Шаг 2).
        """
        # --- Шаг 1: Поиск информации ---
        search_prompt = f"""
        Найдите информацию по следующему запросу, используя поиск Google:
        Запрос: "{query.sub_query}"
        Цель: "{query.purpose}"
        Предоставьте наиболее релевантные результаты поиска.
        """
        await self._log(f"Шаг 1: Выполнение поиска для: '{query.sub_query}' (Цель: {query.purpose})")
        # Вызываем Gemini с инструментом поиска, ожидаем текстовый ответ
        search_results_text = await self._call_gemini(search_prompt, response_schema=None, use_search_tool=True)

        if not isinstance(search_results_text, str) or not search_results_text.strip():
            await self._log(f"Поиск не вернул результатов для запроса: '{query.sub_query}'.")
            return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": "Поиск не дал результатов."}

        await self._log(f"Шаг 1: Получены результаты поиска (длина: {len(search_results_text)}).")

        # --- Шаг 2: Извлечение знаний из результатов поиска ---
        extract_prompt = f"""
        Вы - ИИ-ассистент для извлечения информации.
        Проанализируйте предоставленный ниже текст (результаты поиска Google по запросу "{query.sub_query}" с целью "{query.purpose}") и извлеките ключевые факты, данные и выводы, относящиеся к цели запроса.

        Текст для анализа:
        ```
        {search_results_text}
        ```

        Извлеките информацию и представьте ее в формате JSON.
        """
        # Схема будет добавлена в _call_gemini
        await self._log(f"Шаг 2: Извлечение знаний из результатов поиска для: '{query.sub_query}'")
        # Вызываем Gemini для извлечения JSON, поиск НЕ используем
        response_data = await self._call_gemini(extract_prompt, response_schema=ExtractedKnowledge, use_search_tool=False)

        if isinstance(response_data, dict):
             knowledge = ExtractedKnowledge.model_validate(response_data)
             await self._log(f"Шаг 2: Извлечено знаний: {len(knowledge.key_insights)} ключевых моментов.")
             knowledge_dict = knowledge.model_dump()
             knowledge_dict["original_query"] = query.sub_query
             knowledge_dict["purpose"] = query.purpose
             return knowledge_dict
        else:
            await self._log(f"Шаг 2: Не удалось извлечь знания для запроса: '{query.sub_query}'. Получен ответ: {response_data}")
            return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": "Не удалось извлечь информацию из результатов поиска."}


    async def _reflect_and_plan_next(self, topic: str) -> ReflectionResult:
        """Анализирует собранные знания и планирует следующие шаги."""
        knowledge_summary = "\n".join(
            f"- По запросу '{k.get('original_query', 'N/A')}': {k.get('source_summary', 'N/A')}. Ключевые моменты: {'; '.join(k.get('key_insights', []))}"
            for k in self.all_knowledge
        )
        history_str = ", ".join(self.queries_history)

        prompt = f"""
        Вы - ИИ-ассистент для рефлексии и планирования исследований.
        Ваша задача - проанализировать собранную информацию по теме "{topic}", определить пробелы и спланировать следующие шаги исследования.

        Собранные знания:
        {knowledge_summary if knowledge_summary else "Пока нет собранных знаний."}

        История запросов: {history_str}

        Проанализируйте текущие знания:
        1. Достаточно ли информации для полного ответа на исходный запрос "{topic}"?
        2. Какие ключевые аспекты темы еще не раскрыты или раскрыты недостаточно?
        3. Какие новые вопросы возникли на основе собранной информации?

        На основе анализа сгенерируйте результат в формате JSON.
        - Если информации достаточно, установите is_complete = true и оставьте new_queries пустым.
        - Если информации недостаточно, установите is_complete = false, опишите пробелы в information_gaps и предложите до {self.breadth} новых, конкретных подзапросов в new_queries для их заполнения. Новые запросы не должны повторять запросы из истории.
        """
        # Схема будет добавлена в _call_gemini
        await self._log("Рефлексия над собранными знаниями и планирование следующих шагов...")
        # Запрашиваем JSON, поиск не используем
        response_data = await self._call_gemini(prompt, response_schema=ReflectionResult, use_search_tool=False)

        if isinstance(response_data, dict):
            reflection = ReflectionResult.model_validate(response_data)
            await self._log(f"Результат рефлексии: Завершено={reflection.is_complete}, Новых запросов={len(reflection.new_queries)}")
            return reflection
        else:
             await self._log(f"Не удалось получить результат рефлексии. Получен ответ: {response_data}")
             raise ResearchError("Не удалось получить результат рефлексии.")

    async def _generate_final_report(self, topic: str) -> str:
        """Генерирует финальный отчет на основе всех собранных знаний."""
        # Логика сборки knowledge_details остается прежней
        knowledge_details = ""
        for i, k in enumerate(self.all_knowledge):
            insights = "\n  - ".join(k.get('key_insights', []))
            knowledge_details += (
                f"\nИсточник {i+1} (Результат поиска по запросу: '{k.get('original_query', 'N/A')}')\n"
                f"Резюме источника: {k.get('source_summary', 'N/A')}\n"
                f"Ключевые моменты:\n  - {insights if insights else 'Нет'}\n"
            )

        prompt = f"""
        Вы - ИИ-аналитик и писатель отчетов.
        Ваша задача - написать исчерпывающий, структурированный и хорошо аргументированный отчет в формате Markdown по теме "{topic}", основываясь ИСКЛЮЧИТЕЛЬНО на предоставленных ниже знаниях, извлеченных из веб-поиска.

        Собранные знания из различных источников:
        {knowledge_details if knowledge_details else "Знания не были собраны."}

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
        # Генерируем отчет как обычный текст, поиск не нужен
        report = await self._call_gemini(prompt, response_schema=None, use_search_tool=False, is_report_generation=True)
        if isinstance(report, str):
            await self._log("Финальный отчет сгенерирован.")
            cleaned_report = report.strip().removeprefix("```markdown").removesuffix("```").strip()
            return cleaned_report
        else:
            await self._log("Ошибка: генерация отчета вернула не строку.")
            raise ResearchError("Ошибка при генерации финального отчета.")

    async def research(self, topic: str, log_callback: Callable[[str], Any] | None = None) -> str:
        """
        Основной метод для запуска итеративного исследования.
        """
        self.log_callback = log_callback
        self.all_knowledge = []
        self.queries_history = []

        try:
            await self._log(f"Начало исследования по теме: '{topic}'")
            current_queries = await self._generate_initial_queries(topic)

            for i in range(self.depth):
                await self._log(f"--- Итерация исследования {i + 1}/{self.depth} ---")
                await self._log(f"Запросы для итерации: {[q.sub_query for q in current_queries]}")

                search_tasks = [self._search_and_extract(query) for query in current_queries]
                knowledge_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                new_knowledge_count = 0
                successful_results = 0
                for result in knowledge_results:
                    if isinstance(result, dict):
                        if result.get("key_insights"):
                            self.all_knowledge.append(result)
                            new_knowledge_count += len(result["key_insights"])
                        else:
                             await self._log(f"Нет ключевых моментов для запроса: '{result.get('original_query')}'")
                        successful_results += 1
                    elif isinstance(result, Exception):
                         await self._log(f"Ошибка при поиске/извлечении на итерации {i+1}: {result}")
                    else:
                         await self._log(f"Неожиданный результат поиска/извлечения: {type(result)}")

                await self._log(f"Итерация {i + 1}: Успешно обработано {successful_results}/{len(current_queries)} запросов. Добавлено {new_knowledge_count} новых ключевых моментов.")

                if not self.all_knowledge and i == 0: # Если после первой итерации нет знаний
                    await self._log("Не удалось собрать информацию на первой итерации. Прерывание исследования.")
                    return "Не удалось найти релевантную информацию по вашему запросу."

                if i < self.depth - 1:
                    reflection = await self._reflect_and_plan_next(topic)
                    if reflection.is_complete or not reflection.new_queries:
                        await self._log("Исследование завершено досрочно по результатам рефлексии.")
                        break
                    else:
                        new_unique_queries = []
                        for nq in reflection.new_queries:
                            if nq.sub_query not in self.queries_history:
                                new_unique_queries.append(nq)
                                self.queries_history.append(nq.sub_query)

                        if not new_unique_queries:
                             await self._log("Нет новых уникальных запросов для следующей итерации.")
                             break

                        current_queries = new_unique_queries[:self.breadth]
                        await self._log(f"Планирование итерации {i + 2} с {len(current_queries)} запросами.")
                else:
                     await self._log(f"Достигнута максимальная глубина исследования ({self.depth}).")

            if not self.all_knowledge:
                 await self._log("Не удалось собрать информацию за все итерации.")
                 return "Не удалось найти релевантную информацию по вашему запросу после всех итераций."


            final_report = await self._generate_final_report(topic)
            return final_report

        except ResearchError as e:
            await self._log(f"Критическая ошибка исследования: {e}")
            return f"Ошибка в процессе исследования: {e}"
        except Exception as e:
            logger.exception(f"Непредвиденная ошибка в research: {e}")
            await self._log(f"Непредвиденная ошибка: {e}")
            return f"Произошла непредвиденная ошибка: {e}"
        finally:
            self.log_callback = None
