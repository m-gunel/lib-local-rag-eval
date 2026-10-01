# Стенд замера качества ранжирования lib-local-rag

Методика: Claude Doc «Методика оценки качества ранжирования lib-local-rag»
(https://claude.ai/code/artifact/7c6ccc5e-6890-41e2-959f-cafeb0ad0675).
Стенд живёт отдельно от проекта: lib-local-rag только запускается как демон
и ничем не меняется (кэш байткода отключён, данные демона — в HOME прогона).

## Состав

| Путь | Что это |
|---|---|
| `raw/` | исходники наборов как скачаны (ai360, ZX Bank, ObliQA, FAQ ЦБ, ответы ЦБ кредитным организациям, акты ЦБ, xlsx ЦБ) |
| `work/translate/` | перевод ZX Bank и ObliQA: глоссарий, части EN/RU, пакеты запросов |
| `work/querygen/` | сгенерированные запросы (xlsx, перефразы FAQ, варианты сценариев 2–4, «без ответа») |
| `corpus/` | файлы, которые индексирует демон (собирается `build_corpus.py`) |
| `data/` | `manifest.jsonl`, `queries.jsonl`, `qrels.tsv`, `excluded.jsonl` |
| `runs/<id>/` | прогон: HOME демона, конфиг, логи, `index_state.json`, `responses_<mode>.jsonl` |
| `reports/` | отчёты `report.py` |
| `daemon-venv/` | окружение демона с версиями из `uv.lock` проекта (lancedb 0.38.0, pylance 11.0.0) |
| `.venv/` | окружение стенда (`requirements.txt`) |

## Модули `src/`

| Модуль | Назначение |
|---|---|
| `cbr_dbrnfaq.py` | сбор 521 вопроса кредитных организаций к ЦБ (30 разделов) и ссылок на акты |
| `cbr_faq_pages.py` | актуальные страницы FAQ ЦБ + проверка, что вопрос/ответ на странице есть |
| `prep_translation.py` | единицы перевода ZX Bank / глав ObliQA, выборка вопросов ObliQA |
| `render.py` | md → docx/pptx (pandoc, как оригиналы ZX Bank), pdf (LibreOffice) |
| `indexer_view.py`, `view_corpus.py` | текст файла глазами парсеров индексатора (запускать `daemon-venv/bin/python`) |
| `build_corpus.py` | сборка корпуса и эталона |
| `pool.py` | независимый BM25 (Snowball) для пулов разметки и фрагментов асессорам |
| `judge_prep.py`, `judge_merge.py` | пакеты для асессоров, взвешенная каппа, итоговые оценки |
| `make_daemon_venv.py` | окружение демона по `uv.lock` |
| `lance_probe.py` | состояние индексов прогона (только чтение) |
| `harness.py` | прогон: демон на корпусе → готовность индексов → запросы hybrid/vector/text |
| `metrics.py` | nDCG@10, Hit@1, Hit@10, MRR@10, Recall@10, бутстрэп, ROC-AUC, подбор порога |
| `report.py` | метрики по сценариям и наборам, сравнение типов поиска (гибрид / BM25 / вектор), парное сравнение прогонов |
| `rule_foreign.py` | оценка 0 по правилу для явно чужих пар из выдачи (документ из набора другой тематики) |
| `vector_probe.py` | векторный поиск: приближённый индекс IVF-PQ против точного перебора на готовом индексе прогона (запускать `daemon-venv/bin/python`) |

Тесты: `.venv/bin/python -m pytest -q tests` (метрики сверены с pytrec_eval).

## Порядок

1. Данные (этап 2): скрипты сбора (`cbr_*.py`, `prep_translation.py`) → перевод и
   генерация запросов (агенты) → `build_corpus.py` → `daemon-venv/bin/python
   src/view_corpus.py corpus` → `validate.py --apply` (исключает варианты сценария 3
   без обозначения в тексте эталона). Шаг `validate.py --apply` обязателен после
   каждой пересборки.
2. Прогон (этап 3): `.venv/bin/python src/harness.py run --run-id R6` (индексатор как
   есть) и `--run-id R7 --extra-config "parquet_record_batch_size: 100000000"` (тот же
   код, полный индекс — обход дефекта потери чанков).
3. Дооценка выдачи hybrid: `judge_prep.py postrun --run R6 --run R7 --modes hybrid
   --batch 40` → `src/judge_workflow.js` частями (`a_from`/`a_to`, `b_from`/`b_to`,
   `skip_c: true`, `model_a: "sonnet"`) → `judge_merge.py collect <раунд> <выводы>` →
   `judge_merge.py todo <раунд>` (повтор пропущенных) → `judge_merge.py conflicts` →
   workflow с `c_pids` → `judge_merge.py merge <раунд>`.
4. Отчёт: `report.py --run R6 --compare R7`.

## Решения, которые стоит знать

- Корпус общий для всех наборов (как рабочая папка сотрудника), а не по набору.
- ZX Bank: у каждого документа ровно один формат (docx/pdf/pptx по кругу), дубли убраны.
- ObliQA: своды AML, BRR, климат, CRS, FATCA, разрезаны на главы (глава = файл, txt).
- Акты ЦБ: официальные PDF cbr.ru — сканы или OCR с латиницей вместо кириллицы,
  поэтому текст взят из цифровых выпусков «Вестника Банка России» (вырезаны страницы акта).
- Корпус 400 документов (27.09): к исходным 282 добавлены акты из ссылок в ответах ЦБ,
  «Разъяснения» ЦБ, обзоры банковского сектора, два выпуска xlsx — «соседи» по теме.
- Сценарий 6 «нет ответа» и порог отложены (27.09): запросы и их разметка лежат в
  `work/querygen/` и `work/judge/noanswer/`, в `data/` не входят.
- Индексатор мерится как есть: `src/indexer/indexer.py` проекта — в исходном состоянии
  (попытка исправления дефекта 27.09 откатана). Прогоны R4/R5 шли на изменённом коде и в
  итог не входят.
- В метрики не входят запросы, эталон которых индексатор не читает (3 скана ai360).

## Прогон на другой машине

Репозиторий: `git@github.com:m-gunel/lib-local-rag-eval.git` (приватный). В нём код,
корпус, тест-кейсы, разметка, отчёты и ответы R6 для сравнения; `raw/`, остальные
прогоны и окружения не хранятся (см. `.gitignore`). На машине с индексатором:

1. `git clone git@github.com:m-gunel/lib-local-rag-eval.git && cd lib-local-rag-eval`
2. Окружение стенда: `python3 -m venv .venv && .venv/bin/pip install httpx` (для прогона
   и отчёта больше ничего не нужно).
3. В проекте индексатора: `uv sync --frozen` (его `.venv` и есть интерпретатор демона).
4. Прогон: `LIB_LOCAL_RAG=<путь к проекту> DAEMON_PY=<путь к проекту>/.venv/bin/python
   .venv/bin/python src/harness.py run --run-id W1 --port 8077`
   (`RAG_MODEL_PATH` — если модель лежит не в `<проект>/models/rubert-tiny2_002`).
5. Отчёт: `.venv/bin/python src/report.py --run W1 --compare R6`.
6. Строка «Неоценённых документов в топ-10» > 0 → дооценка: перенести `runs/W1/`
   на машину с Claude и пройти шаги раздела «Порядок», п. 3–4.

Пути документов в ответах считаются от `corpus/`, поэтому прогон с другой машины
читается здесь без правок.
