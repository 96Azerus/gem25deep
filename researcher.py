# researcher.py (v1.2.2 - Fixed google.api_core import)
import asyncio
import json
import logging
from typing import List, Dict, Any, Callable, Coroutine, Optional

# --- ИЗМЕНЕНИЯ ИМПОРТОВ ---
from google import genai
from google.genai import types as genai_types
# УБИРАЕМ импорт google.api_core
# from google.api_core import exceptions as google_api_exceptions
# ИМПОРТИРУЕМ ошибки из genai
from google.genai import errors as genai_errors
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

# --- Основной класс ---
class DeepResearcher:
    def __init__(
        self,
        model_name: str,
        depth: int = 2,
        breadth: int = 3,
        max_completion_tokens: int = 8192,
        temperature: float = 0.3,
        request_timeout: int = 300
    ):
        self.depth = depth
        self.breadth = breadth
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout

        # --- Инициализация клиента ---
        try:
            self.client = genai.Client() # Ключ из GOOGLE_API_KEY
            self.client.models.list(page_size=1)
            logger.info(f"Клиент Google GenAI SDK для DeepResearcher инициализирован (модель по умолчанию: {self.model_name}).")
        # --- ИЗМЕНЕНИЯ В ОБРАБОТКЕ ОШИБОК ---
        except genai_errors.PermissionDenied as e: # Используем genai_errors
             logger.exception(f"Ошибка прав доступа при инициализации клиента GenAI: {e}")
             raise ResearchError(f"Не удалось настроить Google GenAI SDK: Неверный API ключ или нет прав доступа. {e}") from e
        except Exception as e:
             logger.exception(f"Ошибка инициализации клиента Google GenAI SDK: {e}")
             raise ResearchError(f"Не удалось настроить Google GenAI SDK: {e}") from e

        self.all_knowledge: List[Dict[str, Any]] = []
        self.queries_history: List[str] = []
        self.log_callback: Optional[Callable[[str], Any]] = None

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
        response_schema: Optional[type[BaseModel]] = None,
        use_search_tool: bool = False,
    ) -> str | Dict[str, Any]:
        """
        Выполняет вызов Google GenAI API с обработкой ошибок и парсингом JSON.
        """
        full_model_name = f'models/{self.model_name}'

        config_dict = {
            'temperature': self.temperature,
            'max_output_tokens': self.max_completion_tokens
        }
        tools_list = []

        if use_search_tool:
            if response_schema:
                await self._log(f"Предупреждение: Поиск GoogleSearch не может использоваться одновременно с запросом JSON схемы для модели {self.model_name}. Поиск отключен.")
            else:
                tools_list.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
                await self._log("Инструмент поиска GoogleSearch активирован.")

        if response_schema:
            config_dict['response_mime_type'] = 'application/json'
            config_dict['response_schema'] = response_schema
            await self._log(f"Запрошен JSON ответ со схемой: {response_schema.__name__}")

        if tools_list:
             config_dict['tools'] = tools_list

        try:
            generation_config = genai_types.GenerateContentConfig(**config_dict)
        except Exception as e:
             await self._log(f"Ошибка создания GenerateContentConfig: {e}")
             raise ResearchError(f"Ошибка конфигурации запроса к Gemini: {e}") from e

        try:
            await self._log(f"Отправка запроса к GenAI (Модель: {full_model_name}, Схема: {response_schema is not None}, Поиск: {use_search_tool})...")

            response = await self.client.aio.models.generate_content(
                model=full_model_name,
                contents=prompt,
                generation_config=generation_config,
                request_options={'timeout': self.request_timeout}
            )

            if not response.candidates:
                 safety_feedback = getattr(response, 'prompt_feedback', None)
                 block_reason = getattr(safety_feedback, 'block_reason', 'Неизвестно')
                 await self._log(f"GenAI вернул пустой ответ. Причина блокировки: {block_reason}. Safety Feedback: {safety_feedback}")
                 raise ResearchError(f"Запрос к GenAI заблокирован (Причина: {block_reason}) или вернул пустой ответ.")

            response_text = response.text

            if response_schema:
                await self._log(f"Получен текстовый ответ для JSON схемы (длина: {len(response_text)}). Попытка парсинга...")
                cleaned_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
                if not cleaned_text:
                     raise ResearchError("GenAI вернул пустой текст при запросе JSON.")

                try:
                    parsed_response = response_schema.model_validate_json(cleaned_text)
                    await self._log("Ответ GenAI успешно распарсен в JSON.")
                    return parsed_response.model_dump()
                except (ValidationError, json.JSONDecodeError) as e:
                    await self._log(f"Ошибка парсинга JSON от GenAI: {e}. Ответ: {cleaned_text[:500]}...")
                    if HAS_JSON_REPAIR:
                        try:
                            await self._log("Попытка исправить JSON с помощью json_repair...")
                            repaired_json_str = json_repair.repair_json(cleaned_text)
                            parsed_response = response_schema.model_validate_json(repaired_json_str)
                            await self._log("JSON успешно исправлен и распарсен.")
                            return parsed_response.model_dump()
                        except Exception as repair_error:
                             await self._log(f"Не удалось исправить JSON: {repair_error}")
                             raise ResearchError(f"Не удалось распарсить или исправить JSON ответ от GenAI: {e}") from e
                    else:
                         raise ResearchError(f"Не удалось распарсить JSON ответ от GenAI (json_repair не найден): {e}") from e
            else:
                if response_text:
                    await self._log(f"Получен текстовый ответ от GenAI (длина: {len(response_text)}).")
                    return response_text
                else:
                    function_calls = getattr(response.candidates[0].content, 'parts', [])
                    fc_parts = [p for p in function_calls if hasattr(p, 'function_call')]
                    if fc_parts:
                         await self._log("GenAI вернул вызов функции, но нет текстового ответа.")
                         return ""
                    else:
                         await self._log("GenAI вернул ответ без текста и без JSON.")
                         raise ResearchError("GenAI вернул пустой текстовый ответ.")

        # --- ИЗМЕНЕНИЯ В ОБРАБОТКЕ ОШИБОК ---
        # Ловим специфичные ошибки из genai.errors, если они там есть и нужны
        except genai_errors.PermissionDenied as e:
            await self._log(f"Ошибка прав доступа при вызове GenAI API: {e}")
            raise ResearchError(f"Ошибка API ключа или прав доступа: {getattr(e, 'message', str(e))}") from e
        except genai_errors.ResourceExhausted as e: # Если такой класс есть в genai.errors
            await self._log(f"Превышена квота GenAI API: {e}")
            raise ResearchError(f"Превышена квота Google API: {getattr(e, 'message', str(e))}") from e
        except genai_errors.InvalidArgument as e: # Если такой класс есть в genai.errors
             await self._log(f"Неверный аргумент при вызове GenAI API: {e}")
             err_message = getattr(e, 'message', str(e))
             # Проверяем текст ошибки на известные проблемы
             if "API key not valid" in err_message:
                  raise ResearchError("Ошибка API ключа Gemini. Проверьте ключ.") from e
             if "JSON mode is not supported" in err_message or "Tool use is not supported" in err_message or "not found for model" in err_message:
                  raise ResearchError(f"Выбранная модель '{self.model_name}' не поддерживает запрошенные функции (JSON, поиск) или не найдена.") from e
             raise ResearchError(f"Неверный аргумент запроса к Google API: {err_message}") from e
        # Ловим базовый класс ошибок API из genai.errors
        except genai_errors.APIError as e:
            await self._log(f"Общая ошибка Google API: {e}")
            raise ResearchError(f"Ошибка при взаимодействии с Google API: {getattr(e, 'message', str(e))}") from e
        # Ловим остальные ошибки
        except Exception as e:
            await self._log(f"Непредвиденная ошибка при вызове GenAI API: {e}")
            logger.exception("Детали непредвиденной ошибки GenAI API:")
            raise ResearchError(f"Непредвиденная ошибка при взаимодействии с GenAI API: {e}") from e

    # --- Методы логики исследования (без изменений в этой части) ---

    async def _generate_initial_queries(self, topic: str) -> List[SearchQuery]:
        prompt = f"""
        You are an AI research planning assistant.
        Your task is to break down the main research topic "{topic}" into {self.breadth} specific, targeted sub-queries.
        Each sub-query should explore a distinct aspect of the topic and have a clear purpose.
        Avoid overly broad or generic queries.
        Provide the result as a JSON object matching the ResearchPlan schema.
        """
        await self._log(f"Generating initial queries for topic: {topic}")
        response_data = await self._call_gemini(
            prompt,
            response_schema=ResearchPlan,
            use_search_tool=False
        )
        if isinstance(response_data, dict):
             try:
                 plan = ResearchPlan.model_validate(response_data)
                 if not plan.query_plan:
                      await self._log("Warning: Gemini returned an empty query plan.")
                      raise ResearchError("Gemini returned an empty query plan.")
                 self.queries_history.extend([q.sub_query for q in plan.query_plan])
                 await self._log(f"Generated {len(plan.query_plan)} initial queries.")
                 return plan.query_plan
             except ValidationError as e:
                  await self._log(f"Pydantic validation error for ResearchPlan: {e}. Data: {response_data}")
                  raise ResearchError("Failed to validate the query plan from Gemini.") from e
        else:
            await self._log(f"Failed to generate initial query plan. Unexpected response type: {type(response_data)}")
            raise ResearchError("Failed to generate initial query plan (invalid response format).")

    async def _search_and_extract(self, query: SearchQuery) -> Dict[str, Any]:
        search_prompt = f"""
        Using the Google Search tool (google_search), find and briefly summarize the most relevant information for the following query:
        Query: "{query.sub_query}"
        Purpose: "{query.purpose}"
        Provide the results as coherent text or a list of key findings.
        """
        await self._log(f"Step 1: Performing search for: '{query.sub_query}' (Purpose: {query.purpose})")
        search_results_text = await self._call_gemini(
            search_prompt,
            response_schema=None,
            use_search_tool=True
        )

        if not isinstance(search_results_text, str) or not search_results_text.strip():
            await self._log(f"Search returned no text results for query: '{query.sub_query}'.")
            return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": "Search yielded no results or text."}

        await self._log(f"Step 1: Search results received (length: {len(search_results_text)}).")

        extract_prompt = f"""
        You are an AI information extraction assistant.
        Analyze the provided text below (results from a Google search for "{query.sub_query}" with the purpose "{query.purpose}") and extract key facts, data, and conclusions relevant to the query's purpose.
        Provide the result as a JSON object matching the ExtractedKnowledge schema.

        Text to analyze:
        ```
        {search_results_text}
        ```
        """
        await self._log(f"Step 2: Extracting knowledge for: '{query.sub_query}'")
        response_data = await self._call_gemini(
            extract_prompt,
            response_schema=ExtractedKnowledge,
            use_search_tool=False
        )

        if isinstance(response_data, dict):
             try:
                 knowledge = ExtractedKnowledge.model_validate(response_data)
                 await self._log(f"Step 2: Extracted {len(knowledge.key_insights)} key insights.")
                 knowledge_dict = knowledge.model_dump()
                 knowledge_dict["original_query"] = query.sub_query
                 knowledge_dict["purpose"] = query.purpose
                 return knowledge_dict
             except ValidationError as e:
                  await self._log(f"Pydantic validation error for ExtractedKnowledge: {e}. Data: {response_data}")
                  return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": f"Failed to validate extracted knowledge: {e}"}
        else:
            await self._log(f"Step 2: Failed to extract knowledge for query: '{query.sub_query}'. Unexpected response type: {type(response_data)}")
            return {"original_query": query.sub_query, "purpose": query.purpose, "key_insights": [], "source_summary": "Failed to extract information (invalid response format)."}

    async def _reflect_and_plan_next(self, topic: str) -> ReflectionResult:
        knowledge_summary = "\n".join(
            f"- Query '{k.get('original_query', 'N/A')}': {k.get('source_summary', 'N/A')}. Insights: {'; '.join(k.get('key_insights', []))}"
            for k in self.all_knowledge if k.get('key_insights')
        )
        history_str = ", ".join(self.queries_history)

        prompt = f"""
        You are an AI assistant for research reflection and planning.
        Your task is to analyze the gathered information on the topic "{topic}", identify gaps, and plan the next research steps.

        Gathered Knowledge:
        {knowledge_summary if knowledge_summary else "No knowledge gathered yet."}

        Query History: {history_str}

        Analyze the current knowledge:
        1. Is there enough information to comprehensively answer the original query "{topic}"?
        2. What key aspects of the topic are still uncovered or insufficiently explored?
        3. What new questions have arisen from the gathered information?

        Based on the analysis, generate the result as a JSON object matching the ReflectionResult schema.
        - If the information is sufficient, set is_complete = true and leave new_queries empty.
        - If insufficient, set is_complete = false, describe the gaps in information_gaps, and propose up to {self.breadth} new, specific sub-queries in new_queries to fill them. New queries should not repeat those in the history.
        """
        await self._log("Reflecting on gathered knowledge and planning next steps...")
        response_data = await self._call_gemini(
            prompt,
            response_schema=ReflectionResult,
            use_search_tool=False
        )

        if isinstance(response_data, dict):
            try:
                reflection = ReflectionResult.model_validate(response_data)
                await self._log(f"Reflection result: Complete={reflection.is_complete}, New Queries={len(reflection.new_queries)}")
                return reflection
            except ValidationError as e:
                 await self._log(f"Pydantic validation error for ReflectionResult: {e}. Data: {response_data}")
                 raise ResearchError("Failed to validate the reflection result from Gemini.") from e
        else:
             await self._log(f"Failed to get reflection result. Unexpected response type: {type(response_data)}")
             raise ResearchError("Failed to get reflection result (invalid response format).")

    async def _generate_final_report(self, topic: str) -> str:
        knowledge_details = ""
        valid_knowledge_count = 0
        for i, k in enumerate(self.all_knowledge):
            if not k.get('key_insights'):
                continue
            valid_knowledge_count += 1
            insights = "\n  - ".join(k.get('key_insights', []))
            knowledge_details += (
                f"\nSource {valid_knowledge_count} (From query: '{k.get('original_query', 'N/A')}')\n"
                f"Summary: {k.get('source_summary', 'N/A')}\n"
                f"Key Insights:\n  - {insights}\n"
            )

        if not knowledge_details:
             await self._log("No valid knowledge gathered to generate a report.")
             return "Insufficient information was gathered to generate a comprehensive report."

        prompt = f"""
        You are an AI analyst and report writer.
        Your task is to write a comprehensive, structured, and well-argued report in Markdown format on the topic "{topic}", based EXCLUSIVELY on the provided knowledge extracted from web searches.

        Gathered Knowledge from Various Sources ({valid_knowledge_count} sources with insights):
        {knowledge_details}

        Report Requirements:
        - Structure: Introduction (context and purpose), Main Body (logically grouped subsections analyzing information from sources), Conclusion (key findings, synthesis of information).
        - Content: Deeply analyze and synthesize information from different sources. Compare perspectives if they differ. Highlight key findings and patterns. Do not just list facts; explain their significance and connections.
        - Language: Objective, analytical, clear, and precise.
        - Citation: DO NOT use source numbers or direct links. All information is considered derived from the provided knowledge.
        - Completeness: Utilize ALL provided knowledge insights.
        - Format: Clean Markdown. Use headings (#, ##, ###), lists (* or -), emphasis (*italic*, **bold**).

        Write the final report.
        """
        await self._log("Generating final report...")
        report = await self._call_gemini(
            prompt,
            response_schema=None,
            use_search_tool=False
        )

        if isinstance(report, str):
            if len(report.strip()) < 50:
                 await self._log(f"Warning: Generated report seems very short (length: {len(report.strip())}).")
            await self._log("Final report generated.")
            cleaned_report = report.strip().removeprefix("```markdown").removesuffix("```").strip()
            return cleaned_report
        else:
            await self._log(f"Error: Report generation returned non-string type: {type(report)}.")
            raise ResearchError("Error during final report generation (invalid response type).")

    async def research(self, topic: str, log_callback: Optional[Callable[[str], Any]] = None) -> str:
        self.log_callback = log_callback
        self.all_knowledge = []
        self.queries_history = []
        start_time = asyncio.get_event_loop().time()

        try:
            await self._log(f"--- Starting research for: '{topic}' (Model: {self.model_name}, Depth: {self.depth}, Breadth: {self.breadth}) ---")
            current_queries = await self._generate_initial_queries(topic)

            if not current_queries:
                 await self._log("Failed to generate initial queries. Aborting research.")
                 return "Error: Could not plan the initial research steps."

            for i in range(self.depth):
                iteration_num = i + 1
                await self._log(f"--- Research Iteration {iteration_num}/{self.depth} ---")
                if not current_queries:
                     await self._log(f"No queries to execute for iteration {iteration_num}. Finishing early.")
                     break

                await self._log(f"Queries for iteration {iteration_num}: {[q.sub_query for q in current_queries]}")

                search_tasks = [self._search_and_extract(query) for query in current_queries]
                knowledge_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                new_knowledge_count = 0
                successful_results = 0
                iteration_errors = []
                for result in knowledge_results:
                    if isinstance(result, dict):
                        successful_results += 1
                        self.all_knowledge.append(result)
                        if result.get("key_insights"):
                            new_knowledge_count += len(result["key_insights"])
                        else:
                             await self._log(f"No key insights found for query: '{result.get('original_query')}'")
                    elif isinstance(result, Exception):
                         error_msg = f"Error during search/extraction in iteration {iteration_num}: {result}"
                         await self._log(error_msg)
                         iteration_errors.append(error_msg)
                    else:
                         await self._log(f"Unexpected result type during search/extraction: {type(result)}")

                await self._log(f"Iteration {iteration_num}: Processed {successful_results}/{len(current_queries)} queries. Added {new_knowledge_count} new key insights.")
                if iteration_errors:
                     await self._log(f"Iteration {iteration_num}: Encountered {len(iteration_errors)} errors.")

                if not self.all_knowledge and iteration_num == 1:
                    await self._log("Failed to gather any information in the first iteration. Aborting.")
                    return "Failed to find relevant information in the initial research phase."

                if iteration_num < self.depth:
                    try:
                        reflection = await self._reflect_and_plan_next(topic)
                        if reflection.is_complete:
                            await self._log("Reflection indicates research is complete. Finishing early.")
                            break
                        elif not reflection.new_queries:
                             await self._log("Reflection yielded no new queries. Finishing early.")
                             break
                        else:
                            new_unique_queries = []
                            for nq in reflection.new_queries:
                                if nq.sub_query not in self.queries_history:
                                    new_unique_queries.append(nq)
                                    self.queries_history.append(nq.sub_query)

                            if not new_unique_queries:
                                 await self._log("No new unique queries generated after reflection. Finishing early.")
                                 break

                            current_queries = new_unique_queries[:self.breadth]
                            await self._log(f"Planning iteration {iteration_num + 1} with {len(current_queries)} new queries.")

                    except ResearchError as reflect_error:
                         await self._log(f"Error during reflection in iteration {iteration_num}: {reflect_error}. Cannot plan next iteration.")
                         break
                else:
                     await self._log(f"Reached maximum research depth ({self.depth}). Proceeding to final report.")

            await self._log("--- Generating Final Report ---")
            if not self.all_knowledge or not any(k.get('key_insights') for k in self.all_knowledge):
                 await self._log("No actionable knowledge gathered across all iterations.")
                 return "Insufficient information was gathered to generate a comprehensive report after all research steps."

            final_report = await self._generate_final_report(topic)
            return final_report

        except ResearchError as e:
            await self._log(f"Critical Research Error: {e}")
            return f"Research Error: {e}"
        except Exception as e:
            logger.exception(f"Unexpected critical error during research: {e}")
            await self._log(f"Unexpected Critical Error: {e}")
            return f"An unexpected server error occurred during the research process: {e}"
        finally:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            await self._log(f"--- Research finished in {duration:.2f} seconds ---")
            self.log_callback = None
