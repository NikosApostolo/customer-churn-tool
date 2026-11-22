import { parseCSV, outcomeOptions, profileFor, prepare, profileSchema, train, ranking, influences, groups } from './engine.mjs';

let historical = null;
let current = null;
let pending = null;
const profiles = new Map();

async function profile(name) {
  if (!profiles.has(name)) {
    const response = await fetch(new URL(`./data/${name}.json`, import.meta.url));
    if (!response.ok) throw new Error('The built-in model could not be loaded. Please try again.');
    profiles.set(name, { ...await response.json(), exported: true });
  }
  return profiles.get(name);
}

function datasetSummary() {
  const from = current;
  const history = from.history;
  const { data, model, ranked, name, mode } = from;
  return {
    name, mode, ranked, winner: model.winner, count: data.X.length, notes: from.notes,
    history: history ? {
      size: history.data.X.length,
      left: history.data.y.reduce((sum, value) => sum + value, 0),
      fields: history.data.schema.map(({ name }) => name),
      metrics: history.model.metrics,
      testSize: history.model.test.length,
    } : null,
  };
}

function recordNotes(data) {
  return [
    ...(data.hasIDs ? [] : ['Customer IDs were generated from row numbers.']),
    ...(data.invalid ? [`${data.invalid.toLocaleString('en')} invalid input values were treated as missing.`] : []),
    ...(data.ignored.length ? [`Columns skipped: ${data.ignored.join(', ')}.`] : []),
    ...(data.inferred ? ['Inputs were detected automatically. Use only fields known before the customer left.'] : []),
  ];
}

function setHistorical(file, name, model, label, mode) {
  const data = prepare(file, { label });
  model ??= train(data);
  const ranked = ranking(data, model, model.test);
  const history = { data, model };
  const next = { data, model, ranked, name, mode, notes: recordNotes(data), history };
  historical = history;
  current = next;
  pending = null;
  return datasetSummary();
}

function scoreActive(file, name, model, history, mode) {
  const data = prepare(file, { schema: profileSchema(model) });
  const notes = recordNotes(data);
  if (history?.data.hasIDs && data.hasIDs) {
    const priorIds = new Set(history.data.ids);
    const overlap = data.ids.reduce((count, id) => count + Number(priorIds.has(id)), 0);
    if (overlap) notes.push(`${overlap.toLocaleString('en')} IDs also appear in the history. Some scores may be from customers used during training.`);
  }
  current = { data, model, ranked: ranking(data, model), name, mode, notes, history };
  pending = null;
  return datasetSummary();
}

async function run({ action, text, name, label, index, field }) {
  switch (action) {
    case 'demo': {
      const response = await fetch(new URL('./data/ibm.csv', import.meta.url));
      if (!response.ok) throw new Error('The demo could not be loaded. Please try again.');
      return setHistorical(parseCSV(await response.text()), 'IBM Telco demo', await profile('ibm'), 'Churn', 'demo');
    }
    case 'inspect': {
      pending = null;
      const file = parseCSV(text);
      if (file.indexes.has('Churn')) return setHistorical(file, name, null, 'Churn', 'historical');
      const options = outcomeOptions(file);
      if (file.rows.length >= 100 && options.length) {
        pending = { file, name, options };
        return { pending: true, options, canScore: Boolean(compatibleHistory(file) || profileFor(file)) };
      }
      return useActiveFile(file, name);
    }
    case 'label': {
      if (!pending) throw new Error('Choose the CSV again to continue.');
      if (label === '') return useActiveFile(pending.file, pending.name);
      if (!pending.options.includes(label)) throw new Error('Choose the column that records who left and who stayed.');
      return setHistorical(pending.file, pending.name, null, label, 'historical');
    }
    case 'details':
      return influences(current.data, current.model, index);
    case 'group':
      return groups(current.history.data, field);
    case 'reset':
      pending = current = historical = null;
      return null;
    default:
      throw new Error('This action is unavailable.');
  }
}

function compatibleHistory(file) {
  return historical && historical.data.schema.every(({ name }) => file.indexes.has(name));
}

async function useActiveFile(file, name) {
  if (compatibleHistory(file)) return scoreActive(file, name, historical.model, historical, 'active-trained');
  const ready = profileFor(file);
  if (!ready) {
    if (historical) {
      const missing = historical.data.schema.filter(({ name }) => !file.indexes.has(name)).map(({ name }) => name);
      throw new Error(`This file needs the same customer inputs as the historical CSV. Missing: ${missing.join(', ')}.`);
    }
    throw new Error('This CSV has no recognized churn outcome and does not match a built-in model. Start with a historical CSV that includes who left and who stayed.');
  }
  const model = await profile(ready);
  return scoreActive(file, name, model, null, ready === 'ibm' ? 'active-ibm' : 'active-2020');
}

self.onmessage = async ({ data: request }) => {
  try {
    const result = await run(request);
    self.postMessage({ id: request.id, result });
  } catch (error) {
    self.postMessage({ id: request.id, error: error.message });
  }
};
