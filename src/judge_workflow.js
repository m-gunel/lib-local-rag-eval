export const meta = {
  name: 'judge-relevance',
  description: 'Разметка релевантности 0–3: первичные асессоры, независимая вторая оценка 20%, третий асессор на расхождениях',
  phases: [
    { title: 'Асессор A', detail: 'все пары' },
    { title: 'Асессор B', detail: '~20% пар, другая разбивка' },
    { title: 'Асессор C', detail: 'расхождения A/B в зоне релевантности, вслепую' },
  ],
}

// args: { round, dir, a: [{file, n}], b: [{file, n}] }
//   или короткая форма: { round, dir, size, a_count, a_last, b_count, b_last }
//   a_from / b_from — начать с этого номера пакета (продолжение прерванного раунда),
//   a_to / b_to — закончить перед этим номером (раунд частями в параллельных запусках)
function expand(prefix, count, last, size, from = 0, to = count) {
  return Array.from({ length: count }, (_, i) => ({
    file: `${prefix}_${String(i).padStart(3, '0')}.json`,
    n: i === count - 1 ? last : size,
  })).slice(from, to)
}
if (!args.a) {
  // a_prefix: пакеты первичной оценки под другим именем (p_/q_ после judge_merge.py rebatch)
  args.a = expand(args.a_prefix || 'a', args.a_count, args.a_last, args.size, args.a_from || 0, args.a_to ?? args.a_count)
  args.b = expand(args.b_prefix || 'b', args.b_count || 0, args.b_last, args.size, args.b_from || 0, args.b_to ?? (args.b_count || 0))
}
const RUBRIC = `Ты — асессор релевантности для оценки поиска по документам в банке. Для каждой пары «информационная потребность — документ» поставь оценку, насколько ДОКУМЕНТ отвечает на потребность:
3 — в показанных фрагментах есть полный ответ;
2 — частичный ответ: есть часть нужных сведений, или ответ нужно дополнить из другого документа;
1 — документ по теме запроса, но ответа в нём нет;
0 — не по теме.
Правила:
- Оценивай только по названию документа и показанным фрагментам; не домысливай, чего в них нет, и не пользуйся внешними знаниями о содержании документа.
- Если потребность про конкретный банк, период, юрисдикцию, форму или акт, а фрагменты про другой — ответа нет (не выше 1), кроме случая, когда нужные сведения прямо есть во фрагментах (например, сравнительные данные за прошлый год в отчёте следующего года).
- Порядок пар в пакете случаен и ничего не значит. Каждую пару оценивай независимо.
- Причина — одно короткое предложение по существу.
- Только читай файл пакета. Ничего не записывай на диск и не создавай файлов — оценки возвращай в ответе.`

const SCHEMA = { type: 'object', properties: { grades: { type: 'array', items: { type: 'object', properties: { pid: { type: 'string' }, grade: { type: 'integer', minimum: 0, maximum: 3 }, reason: { type: 'string' } }, required: ['pid', 'grade', 'reason'] } } }, required: ['grades'] }

function batchPrompt(file, n) {
  return `${RUBRIC}
Прочитай файл ${args.dir}/${file} — JSON-массив из ${n} пар {pid, qid, need, doc_id, title, passages}. need — информационная потребность пользователя, passages — фрагменты документа. Оцени ВСЕ ${n} пар и верни для каждой pid, оценку и причину.`
}

// Модель асессора: args.model_a / args.model_b (или общий args.model); без
// указания — модель сессии. C всегда на модели сессии.
async function runBatches(list, label, ph) {
  const model = label === 'C' ? null : (label === 'A' ? args.model_a : args.model_b) || args.model
  const res = await parallel(list.map(b => () =>
    agent(batchPrompt(b.file, b.n), { label: `${label} ${b.file}`, phase: ph, schema: SCHEMA, ...(model ? { model } : {}) })
      .then(r => {
        if (r && r.grades.length !== b.n) log(`${b.file}: оценок ${r.grades.length} из ${b.n}`)
        return r ? r.grades : []
      })))
  return res.filter(Boolean).flat()
}

// Режим «только C» по готовым пакетам c_XXX.json (judge_merge.py → пары с расхождением A/B).
if (args.c_count) {
  const list = expand('c', args.c_count, args.c_last, args.size)
  const C = await runBatches(list, 'C', 'Асессор C')
  return { round: args.round, A: [], B: [], C, conflicts: [] }
}
// Режим «только C»: args.c_pids — пары с расхождением A/B, собранные по всем запускам раунда.
if (args.c_pids) {
  const groups = []
  for (let i = 0; i < args.c_pids.length; i += 20) groups.push(args.c_pids.slice(i, i + 20))
  const C = (await parallel(groups.map((g, i) => () => agent(`${RUBRIC}
Пары лежат в файлах ${args.dir}/a_*.json (JSON-массивы пар {pid, qid, need, doc_id, title, passages}). Найди и оцени ТОЛЬКО пары с этими pid: ${g.join(', ')}. Других оценок ты не видишь и не ищешь — оцени сам. Верни для каждой pid оценку и причину.`,
    { label: `C ${i}`, phase: 'Асессор C', schema: SCHEMA }).then(r => r ? r.grades.filter(x => g.includes(x.pid)) : [])))).filter(Boolean).flat()
  return { round: args.round, A: [], B: [], C, conflicts: args.c_pids }
}

const [A, B] = await Promise.all([
  runBatches(args.a, 'A', 'Асессор A'),
  runBatches(args.b, 'B', 'Асессор B'),
])
// Раунд идёт несколькими запусками: расхождения считаются по всем оценкам
// сразу (judge_merge.py conflicts), а третий асессор — отдельным запуском.
if (args.skip_c) return { round: args.round, A, B, C: [], conflicts: [] }

const a = new Map(A.map(g => [g.pid, g.grade]))
const conflicts = B.filter(g => a.has(g.pid) && a.get(g.pid) !== g.grade && Math.max(a.get(g.pid), g.grade) >= 2).map(g => g.pid)
log(`двойная оценка: ${B.length}, расхождений в зоне релевантности: ${conflicts.length}`)

let C = []
if (conflicts.length) {
  const groups = []
  for (let i = 0; i < conflicts.length; i += 20) groups.push(conflicts.slice(i, i + 20))
  C = (await parallel(groups.map((g, i) => () => agent(`${RUBRIC}
Пары лежат в файлах ${args.dir}/a_*.json (JSON-массивы пар {pid, qid, need, doc_id, title, passages}). Найди и оцени ТОЛЬКО пары с этими pid: ${g.join(', ')}. Других оценок ты не видишь и не ищешь — оцени сам. Верни для каждой pid оценку и причину.`,
    { label: `C ${i}`, phase: 'Асессор C', schema: SCHEMA }).then(r => r ? r.grades.filter(x => g.includes(x.pid)) : [])))).filter(Boolean).flat()
}
return { round: args.round, A, B, C, conflicts }
