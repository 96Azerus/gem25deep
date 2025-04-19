# researcher.py (v1.1.0)
import asyncio
import json
import logging
from typing import List, Dict, Any, Callable, Coroutine

import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# --- Модели Pydantic для структурированного вывода ---

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

# --- Класс ошибки исследования ---

class ResearchError(Exception):
    """Пользовательское исключение для ошибок в процессе исследования."""
    pass

# --- Основной класс исследователя ---

class DeepResearcher:
    """
    Агент для глубокого итеративного исследования с использованием Gemini API.
    """
    def __init__(
        self,
        api_key: str, # Ключ API теперь обязателен
        model_name: str, # Имя модели теперь обязательный параметр
        depth: int = 2,
        breadth: int = 3,
        max_completion_tokens: int = 8192,
        temperature: float = 0.3,
    ):
        """
        Инициализация исследователя.

        Args:
            api_key: API ключ для Google Gemini.
            model_name: Имя модели Gemini для использования.
            depth: Максимальное количество итераций углубления.
            breadth: Количество поисковых запросов на каждой итерации.
            max_completion_tokens: Максимальное количество токенов в ответе модели.
            temperature: Температура для генерации (контроль креативности).
        """
        self.depth = depth
        self.breadth = breadth
        self.model_name = model_name # Сохраняем имя модели
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.api_key = api_key # Сохраняем ключ

        # Конфигурируем API ключ при инициализации
        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
             logger.exception(f"Ошибка конфигурации Gemini API ключа: {e}")
             raise ResearchError(f"Не удалось настроить Gemini API ключ: {e}") from e

        # Инициализируем модели
        try:
            # Основная модель для планирования, извлечения, рефлексии
            self.model = genai.GenerativeModel(
                self.model_name,
                tools=['google_search_retrieval'],
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                )
            )
            # Модель для генерации финального отчета (может быть той же)
            self.report_model = genai.GenerativeModel(
                 self.model_name,
                 tools=['google_search_retrieval'],
                 generation_config=genai.GenerationConfig(
                    max_output_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                 )
            )
            logger.info(f"Модель Gemini '{self.model_name}' успешно инициализирована.")
        except Exception as e:
            logger.exception(f"Ошибка инициализации модели Gemini '{self.model_name}': {e}")
            raise ResearchError(f"Не удалось инициализировать модель Gemini '{self.model_name}': {e}") from e

        self.all_knowledge: List[Dict[str, Any]] = []
        self.queries_history: List[str] = []
        self.log_callback: Callable[[str], Any] | None = None # Тип колбэка изменен

    async def _log(self, message: str):
        """Асинхронно вызывает лог-колбэк, если он установлен."""
        logger.info(message)
        if self.log_callback:
            try:
                # Проверяем, является ли колбэк корутиной
                if asyncio.iscoroutinefunction(self.log_callback):
                    asyncio.create_task(self.log_callback(message))
                else:
                    # Если это обычная функция, вызываем ее напрямую
                    # (или в executor'е, если она блокирующая и это критично)
                    self.log_callback(message)
            except Exception as e:
                logger.error(f"Ошибка при вызове log_callback: {e}")


    async def _call_gemini(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None, # Используем type[BaseModel]
        is_report_generation: bool = False
    ) -> str | Dict[str, Any]:
        """
        Выполняет вызов Gemini API с обработкой ошибок и парсингом JSON.
        """
        model_to_use = self.report_model if is_report_generation else self.model
        generation_config_override = None
        if response_schema:
            # Устанавливаем JSON mode только если нужна схема
            generation_config_override = genai.GenerationConfig(response_mime_type="application/json")
            # Добавляем описание схемы в промпт, как требует Gemini API
            prompt += f"\n\nПожалуйста, отформатируйте ваш ответ как JSON объект, соответствующий следующей Pydantic схеме:\n```json\n{response_schema.model_json_schema(indent=2)}\n```"
            prompt += "\nВаш ответ должен содержать ТОЛЬКО валидный JSON объект без каких-либо других текстовых пояснений до или после него."


        try:
            await self._log(f"Отправка запроса к Gemini (Модель: {model_to_use.model_name}, Схема: {response_schema is not None}, Отчет: {is_report_generation})...")

            response = await model_to_use.generate_content_async(
                prompt,
                # Передаем tool_config только если инструменты действительно нужны для этого вызова
                # В данном случае, поиск нужен всегда, кроме генерации отчета по уже собранным данным
                tool_config={'google_search_retrieval': {'mode': 'AUTO'}}, # Явно включаем поиск
                generation_config=generation_config_override,
                request_options={'timeout': 300}
            )

            if not response.candidates or not response.candidates[0].content.parts:
                 await self._log("Gemini вернул пустой ответ.")
                 # Попробуем получить safety ratings для диагностики
                 try:
                     safety_info = response.prompt_feedback
                     await self._log(f"Safety Feedback: {safety_info}")
                 except Exception:
                     pass # Игнорируем ошибки при получении safety ratings
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
                    # Попытка исправить JSON, если он немного некорректен
                    try:
                        import json_repair
                        repaired_json_str = json_repair.repair_json(cleaned_text)
                        parsed_response = response_schema.model_validate_json(repaired_json_str)
                        await self._log("JSON успешно исправлен и распарсен.")
                        return parsed_response.model_dump()
                    except Exception as repair_error:
                         await self._log(f"Не удалось исправить JSON: {repair_error}")
                         raise ResearchError(f"Не удалось распарсить JSON ответ от Gemini: {e}") from e
            else:
                return response_text

        except Exception as e:
            await self._log(f"Ошибка при вызове Gemini API: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
                 await self._log(f"Safety Feedback: {e.response.prompt_feedback}")
                 if e.response.prompt_feedback.block_reason:
                     raise ResearchError(f"Запрос к Gemini заблокирован: {e.response.prompt_feedback.block_reason}") from e
            if "API key not valid" in str(e):
                 raise ResearchError("Ошибка API ключа Gemini. Проверьте ключ.") from e
            elif "quota" in str(e).lower():
                 raise ResearchError("Превышена квота Gemini API.") from e
            raise ResearchError(f"Ошибка при взаимодействии с Gemini API: {e}") from e

    async def _generate_initial_queries(self, topic: str) -> List[SearchQuery]:
        """Генерирует начальные поисковые запросы."""
        prompt = f"""
        Вы - ИИ-ассистент для планирования исследований.
        Ваша задача - разбить основную тему исследования "{topic}" на {self.breadth} конкретных, целенаправленных поисковых подзапросов.
        Каждый подзапрос должен исследовать отдельный аспект темы и иметь четкую цель.
        Избегайте слишком широких или общих запросов.
        """
        # Схема добавляется в _call_gemini
        await self._log(f"Генерация начальных запросов для темы: {topic}")
        response_data = await self._call_gemini(prompt, response_schema=ResearchPlan)
        if isinstance(response_data, dict) and "query_plan" in response_data:
             plan = ResearchPlan.model_validate(response_data)
             self.queries_history.extend([q.sub_query for q in plan.query_plan])
             await self._log(f"Сгенерировано {len(plan.query_plan)} начальных запросов.")
             return plan.query_plan
        else:
            await self._log("Не удалось сгенерировать начальный план запросов.")
            raise ResearchError("Не удалось сгенерировать начальный план запросов.")


    async def _search_and_extract(self, query: SearchQuery) -> Dict[str, Any]:
        """Выполняет поиск по подзапросу и извлекает знания."""
        # Промпт для Gemini, чтобы он использовал поиск и извлек информацию
        prompt_extract = f"""
        Проведите исследование по следующему подзапросу, связанному с основной темой:
        Подзапрос: "{query.sub_query}"
        Цель этого подзапроса: "{query.purpose}"

        Используя встроенный инструмент поиска Google, найдите наиболее релевантную информацию.
        Затем проанализируйте найденную информацию и извлеките ключевые факты, данные и выводы, относящиеся к цели запроса.

        Представьте результат в формате JSON, используя следующую Pydantic схему:
        {ExtractedKnowledge.model_json_schema(indent=2)}

        Ваш ответ должен содержать ТОЛЬКО валидный JSON объект.
        """
        await self._log(f"Выполнение поиска и извлечения для: '{query.sub_query}' (Цель: {query.purpose})")
        # Gemini сам выполнит поиск и вернет результат, из которого мы попросили извлечь JSON
        response_data = await self._call_gemini(prompt_extract, response_schema=ExtractedKnowledge)

        if isinstance(response_data, dict):
             knowledge = ExtractedKnowledge.model_validate(response_data)
             await self._log(f"Извлечено знаний: {len(knowledge.key_insights)} ключевых моментов.")
             knowledge_dict = knowledge.model_dump()
             knowledge_dict["original_query"] = query.sub_query
             knowledge_dict["purpose"] = query.purpose
             return knowledge_dict
        else:
            await self._log(f"Не удалось извлечь знания для запроса: '{query.sub_query}'.")
            return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": "Не удалось извлечь информацию."}


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
        """
        # Схема добавляется в _call_gemini
        await self._log("Рефлексия над собранными знаниями и планирование следующих шагов...")
        response_data = await self._call_gemini(prompt, response_schema=ReflectionResult)

        if isinstance(response_data, dict):
            reflection = ReflectionResult.model_validate(response_data)
            await self._log(f"Результат рефлексии: Завершено={reflection.is_complete}, Новых запросов={len(reflection.new_queries)}")
            return reflection
        else:
             await self._log("Не удалось получить результат рефлексии.")
             raise ResearchError("Не удалось получить результат рефлексии.")

    async def _generate_final_report(self, topic: str) -> str:
        """Генерирует финальный отчет на основе всех собранных знаний."""
        knowledge_details = ""
        for i, k in enumerate(self.all_knowledge):
            insights = "\n  - ".join(k.get('key_insights', []))
            knowledge_details += (
                f"\nИсточник {i+1} (Результат поиска по запросу: '{k.get('original_query', 'N/A')}')\n"
                # f"Цель запроса: '{k.get('purpose', 'N/A')}'\n" # Можно убрать цель из финального отчета
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
        - Полнота: Постарайтесь использовать всю релевантную информацию из предоставленных знаний.
        - Формат: Чистый Markdown. Используйте заголовки (#, ##, ###), списки (* или -), выделение текста (*курсив*, **жирный**).

        Напишите финальный отчет.
        """
        await self._log("Генерация финального отчета...")
        # Используем report_model для генерации отчета
        report = await self._call_gemini(prompt, is_report_generation=True)
        if isinstance(report, str):
            await self._log("Финальный отчет сгенерирован.")
            # Убираем возможные Markdown обертки ```markdown ... ```
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
                        self.all_knowledge.append(result)
                        if result.get("key_insights"):
                             new_knowledge_count += len(result["key_insights"])
                        successful_results += 1
                    elif isinstance(result, Exception):
                         await self._log(f"Ошибка при поиске/извлечении на итерации {i+1}: {result}")
                    else:
                         await self._log(f"Неожиданный результат поиска/извлечения: {type(result)}")


                await self._log(f"Итерация {i + 1}: Успешно обработано {successful_results}/{len(current_queries)} запросов. Добавлено {new_knowledge_count} новых ключевых моментов.")

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