const IBM_NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'];
const IBM_CATEGORIES = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'];
const TELECOM_NUMERIC = ['account_length', 'number_vmail_messages', 'number_customer_service_calls',
  ...['day', 'eve', 'night', 'intl'].flatMap(period => ['minutes', 'calls', 'charge'].map(unit => `total_${period}_${unit}`))];
const TELECOM_CATEGORIES = ['state', 'area_code', 'international_plan', 'voice_mail_plan'];
const NONNEGATIVE = new Set([...IBM_NUMERIC, ...TELECOM_NUMERIC]);
const NUMERIC = [...NONNEGATIVE];
const CATEGORIES = [...IBM_CATEGORIES, ...TELECOM_CATEGORIES];
const fields = new Map([...NUMERIC, ...CATEGORIES, 'Churn', 'customerID'].map(name => [key(name), name]));
for (const alias of ['churned', 'is_churn', 'is_churned', 'churn_flag', 'churn_label', 'customer_churn', 'attrition', 'exited']) fields.set(key(alias), 'Churn');
for (const alias of ['id', 'customer_id', 'customer id']) fields.set(key(alias), 'customerID');
const targets = new Map([
  ...['yes', '1', '1.0', 'true', 'churned', 'left'].map(value => [value, 1]),
  ...['no', '0', '0.0', 'false', 'retained', 'stayed'].map(value => [value, 0]),
]);
const excluded = /(?:^|[^a-z0-9])(id|name|email|address|phone|churn|attrition|target|label|outcome|status|exit|cancellation|retention|probability|prediction|score|reason)(?:$|[^a-z0-9])/i;
const absent = new Set(['', 'na', 'n/a', 'null', 'none', 'nan']);

function key(name) {
  return name.toLowerCase().replace(/[^a-z0-9]/g, '');
}

function number(value) {
  return /^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(value) && Math.abs(Number(value)) <= 1e100
    ? Number(value) : null;
}

function rawValue(value) {
  const cleaned = value.trim().toLowerCase();
  return absent.has(cleaned) ? null : cleaned;
}

function records(text, separator, limit = Infinity) {
  const rows = [];
  let row = [], value = '', quoted = false, closed = false;
  function finishValue() {
    row.push(value);
    value = '';
    closed = false;
  }
  function finishRow() {
    finishValue();
    if (row.some(cell => cell.trim() !== '') || row.length > 1) rows.push(row);
    row = [];
  }
  for (let i = 0; i < text.length && rows.length < limit; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        value += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
        closed = true;
      } else {
        value += ch;
      }
    } else if (ch === separator) {
      finishValue();
    } else if (ch === '\r' || ch === '\n') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      finishRow();
    } else if (ch === '"' && !value.trim() && !closed) {
      value = '';
      quoted = true;
    } else if (!closed || !ch.trim()) {
      value += ch;
    } else {
      throw new Error('Check the quotation marks in your CSV.');
    }
  }
  if (quoted) throw new Error('A quoted CSV field was left open.');
  if (rows.length < limit && (value.length || row.length || closed)) finishRow();
  return rows;
}

export function parseCSV(text) {
  text = text.replace(/^\uFEFF/, '');
  if (!text.trim()) throw new Error('The CSV is empty.');
  let separator = ',', width = 0;
  for (const candidate of [',', ';', '\t', '|']) {
    try {
      const count = records(text, candidate, 1)[0]?.length ?? 0;
      if (count > width) {
        width = count;
        separator = candidate;
      }
    } catch {}
  }
  if (width < 2) throw new Error('Include at least two columns with a header. Use a comma, semicolon, tab or pipe separator.');
  const [header, ...rows] = records(text, separator, 15002);
  const names = header.map(name => fields.get(key(name)) ?? name.trim());
  if (names.some(name => !name) || new Set(names.map(name => name.toLowerCase())).size !== names.length) {
    throw new Error('Give every column a unique, nonempty header.');
  }
  if (!rows.length || rows.length > 15000) throw new Error('Use a CSV with 1 to 15,000 customer rows.');
  const mismatch = rows.findIndex(row => row.length !== names.length);
  if (mismatch >= 0) throw new Error(`Data row ${mismatch + 1} has a different number of fields than the header.`);
  return { names, rows, indexes: new Map(names.map((name, index) => [name, index])) };
}

export function outcomeOptions(file) {
  return file.names.filter((name, index) => name !== 'customerID' && name !== 'Churn'
    && !NUMERIC.includes(name) && !CATEGORIES.includes(name)
    && file.rows.every(row => targets.has(row[index].trim().toLowerCase()))
    && new Set(file.rows.map(row => targets.get(row[index].trim().toLowerCase()))).size === 2);
}

export function profileFor(file) {
  if ([...TELECOM_NUMERIC, ...TELECOM_CATEGORIES].every(name => file.indexes.has(name))) return 'telecom2020';
  if ([...IBM_NUMERIC, ...IBM_CATEGORIES].every(name => file.indexes.has(name))) return 'ibm';
  return null;
}

export function prepare(file, { label, schema } = {}) {
  const { rows, names, indexes } = file;
  const indexOfId = indexes.get('customerID');
  const ids = rows.map((row, index) => indexOfId === undefined ? `ROW-${index + 1}` : row[indexOfId].trim());
  if (ids.some(id => !id) || new Set(ids).size !== ids.length) {
    throw new Error('Every customer ID must be filled and unique.');
  }
  const known = [...NUMERIC, ...CATEGORIES].filter(name => name !== label && indexes.has(name));
  if (!schema) {
    schema = known.map(name => ({ name, numeric: NONNEGATIVE.has(name) }));
    if (!schema.length) {
      for (const [index, name] of names.entries()) {
        if (name === label || name === 'Churn' || name === 'customerID' || excluded.test(name)) continue;
        const values = rows.map(row => rawValue(row[index])).filter(value => value !== null);
        if (!values.length) continue;
        const numeric = values.every(value => number(value) !== null);
        if (numeric || new Set(values).size <= Math.min(50, Math.max(2, Math.floor(values.length / 5)))) {
          schema.push({ name, numeric });
        }
      }
    }
  }
  if (!schema.length) throw new Error('Include customer features recorded before churn, such as tenure or contract.');
  const missing = schema.filter(({ name }) => !indexes.has(name)).map(({ name }) => name);
  if (missing.length) throw new Error(`Add the missing input columns: ${missing.join(', ')}.`);
  const inputs = schema.map(field => ({ ...field, index: indexes.get(field.name) }));
  let invalid = 0;
  const X = rows.map(row => inputs.map(({ name, index, numeric }) => {
    const raw = rawValue(row[index]);
    if (raw === null || !numeric) return raw;
    const value = number(raw);
    if (value === null || (NONNEGATIVE.has(name) && value < 0) || (name === 'SeniorCitizen' && value !== 0 && value !== 1)) {
      invalid++;
      return null;
    }
    return value;
  }));
  for (const [i, field] of schema.entries()) {
    if (!field.numeric && new Set(X.map(row => row[i]).filter(value => value !== null)).size > (field.name === 'state' ? 51 : 50)) {
      throw new Error(`“${field.name}” has too many categories. Remove identifiers or free-text columns.`);
    }
  }
  const ignored = names.filter(name => name !== 'customerID' && name !== label && !schema.some(field => field.name === name));
  let y = null;
  if (label) {
    if (rows.length < 100) throw new Error('To learn from outcomes, include at least 100 historical customers.');
    y = rows.map(row => targets.get(row[indexes.get(label)].trim().toLowerCase()));
    if (y.includes(undefined)) throw new Error('Use Yes/No, True/False, 1/0 or Left/Stayed for every churn outcome.');
    const count = y.reduce((sum, value) => sum + value, 0);
    if (count < 20 || y.length - count < 20) throw new Error('Include at least 20 customers who left and 20 who stayed.');
  }
  return { X, y, ids, schema, invalid, ignored, inferred: !known.length, hasIDs: indexOfId !== undefined };
}

export function profileSchema(profile) {
  return profile.columns.map(column => ({ name: column.name, numeric: !column.categories }));
}

function quantile(sorted, location) {
  const pos = (sorted.length - 1) * location;
  const before = Math.floor(pos);
  return sorted.length ? sorted[before] + (sorted[Math.ceil(pos)] - sorted[before]) * (pos - before) : 0;
}

function fitColumns(data, indices) {
  return data.schema.map(({ name, numeric }, i) => {
    const values = indices.map(index => data.X[index][i]);
    if (numeric) {
      const sorted = values.filter(value => value !== null).sort((a, b) => a - b);
      const median = quantile(sorted, 0.5);
      const mean = values.reduce((sum, value) => sum + (value ?? median), 0) / values.length;
      const variance = values.reduce((sum, value) => sum + ((value ?? median) - mean) ** 2, 0) / values.length;
      return { name, median, mean, scale: Math.sqrt(variance) || 1 };
    }
    const counts = new Map();
    for (const value of values) counts.set(value ?? 'missing', (counts.get(value ?? 'missing') ?? 0) + 1);
    const categories = [...counts.keys()].sort();
    const reference = categories.reduce((best, value) => counts.get(value) > counts.get(best) ? value : best);
    return { name, categories, reference };
  });
}

function encode(row, columns) {
  const vector = [];
  for (const [i, column] of columns.entries()) {
    if (column.categories) {
      for (const category of column.categories) vector.push(Number((row[i] ?? 'missing') === category));
    } else {
      vector.push(((row[i] ?? column.median) - column.mean) / column.scale);
    }
  }
  return vector;
}

function random(seed) {
  return () => {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

function shuffled(values, next) {
  const copy = [...values];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function splits(y) {
  const next = random(42);
  const train = [], val = [], test = [];
  for (const label of [0, 1]) {
    const group = shuffled(y.flatMap((value, i) => value === label ? [i] : []), next);
    const stopTrain = Math.floor(group.length * .6);
    const stopVal = Math.floor(group.length * .8);
    train.push(...group.slice(0, stopTrain));
    val.push(...group.slice(stopTrain, stopVal));
    test.push(...group.slice(stopVal));
  }
  return { train: shuffled(train, next), val: shuffled(val, next), test: shuffled(test, next) };
}

const sigmoid = value => 1 / (1 + Math.exp(-value));
const logit = value => Math.log(value / (1 - value));
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

function logistic(X, y) {
  let base = logit(mean(y));
  const weights = new Float64Array(X[0].length);
  const sparse = X.map(row => row.flatMap((value, index) => value === 0 ? [] : [[index, value]]));
  for (let round = 0; round < 115; round++) {
    const gradient = new Float64Array(weights.length);
    let bias = 0;
    for (let i = 0; i < sparse.length; i++) {
      let score = base;
      for (const [j, value] of sparse[i]) score += weights[j] * value;
      const error = sigmoid(score) - y[i];
      bias += error;
      for (const [j, value] of sparse[i]) gradient[j] += error * value;
    }
    const step = .45 / (1 + round / 95);
    base -= step * bias / X.length;
    for (let j = 0; j < weights.length; j++) weights[j] -= step * (gradient[j] + weights[j]) / X.length;
  }
  return { kind: 'logistic', base, weights };
}

function buildBins(X) {
  return X[0].map((_, feature) => {
    const values = X.map(row => row[feature]);
    const sorted = [...values].sort((a, b) => a - b);
    const unique = [...new Set(sorted)];
    const thresholds = unique.length <= 10
      ? unique.slice(1)
      : [...new Set(Array.from({ length: 8 }, (_, i) => quantile(sorted, (i + 1) / 9)))]
        .filter(value => value > sorted[0]);
    const codes = Uint8Array.from(values, value => {
      const firstAbove = thresholds.findIndex(edge => edge > value);
      return firstAbove < 0 ? thresholds.length : firstAbove;
    });
    return { codes, thresholds };
  });
}

function grow(indices, depth, features, bins, gradient, hessian) {
  const penalty = 5;
  let G = 0, H = 0;
  for (const index of indices) {
    G += gradient[index];
    H += hessian[index];
  }
  const weight = .04 * G / (H + penalty);
  if (!depth || indices.length < 16) return weight;
  let best = null;
  let bestGain = 0;
  const root = G * G / (H + penalty);
  for (const feature of features) {
    const { codes, thresholds } = bins[feature];
    if (!thresholds.length) continue;
    const g = new Float64Array(thresholds.length + 1);
    const h = new Float64Array(thresholds.length + 1);
    const counts = new Uint32Array(thresholds.length + 1);
    for (const index of indices) {
      const code = codes[index];
      g[code] += gradient[index];
      h[code] += hessian[index];
      counts[code]++;
    }
    let leftG = 0, leftH = 0, count = 0;
    for (let i = 0; i < thresholds.length; i++) {
      leftG += g[i];
      leftH += h[i];
      count += counts[i];
      if (count < 8 || indices.length - count < 8) continue;
      const gain = leftG ** 2 / (leftH + penalty) + (G - leftG) ** 2 / (H - leftH + penalty) - root;
      if (gain > bestGain) {
        bestGain = gain;
        best = [feature, i];
      }
    }
  }
  if (!best) return weight;
  const [feature, split] = best;
  const { codes, thresholds } = bins[feature];
  const left = [], right = [];
  for (const index of indices) (codes[index] <= split ? left : right).push(index);
  return [feature, thresholds[split],
    grow(left, depth - 1, features, bins, gradient, hessian),
    grow(right, depth - 1, features, bins, gradient, hessian)];
}

function treeValue(tree, vector, float32) {
  while (Array.isArray(tree)) {
    const value = vector[tree[0]];
    tree = (float32 ? Math.fround(value) < Math.fround(tree[1]) : value < tree[1]) ? tree[2] : tree[3];
  }
  return tree;
}

function boost(X, y) {
  const base = logit(mean(y));
  const scores = new Float64Array(y.length).fill(base);
  const bins = buildBins(X);
  const candidates = bins.flatMap(({ thresholds }, i) => thresholds.length ? [i] : []);
  const indices = Array.from(y.keys());
  const next = random(91), trees = [];
  for (let round = 0; round < 180; round++) {
    const probabilities = scores.map(sigmoid);
    const gradient = probabilities.map((p, i) => y[i] - p);
    const hessian = probabilities.map(p => p * (1 - p));
    const features = shuffled(candidates, next).slice(0, 20);
    const tree = grow(indices, 2, features, bins, gradient, hessian);
    trees.push(tree);
    for (let i = 0; i < scores.length; i++) scores[i] += treeValue(tree, X[i], false);
  }
  return { kind: 'boosted', base, trees };
}

function probability(estimator, vector, float32) {
  let total = estimator.base;
  if (estimator.kind === 'logistic') {
    for (const [i, weight] of estimator.weights.entries()) total += weight * vector[i];
  } else {
    for (const tree of estimator.trees) total += treeValue(tree, vector, float32);
  }
  return sigmoid(total);
}

export function score(model, row) {
  return probability(model.estimator, encode(row, model.columns), model.exported === true);
}

function averagePrecision(truth, scores) {
  const sorted = truth.map((value, index) => [scores[index], value]).sort((a, b) => b[0] - a[0]);
  let positive = 0, lastPositive = 0, precision = 0;
  for (let i = 0; i < sorted.length; i++) {
    positive += sorted[i][1];
    if (i === sorted.length - 1 || sorted[i][0] !== sorted[i + 1][0]) {
      precision += (positive - lastPositive) * positive / (i + 1);
      lastPositive = positive;
    }
  }
  return precision / positive;
}

function evaluate(name, truth, scores) {
  const matrix = [[0, 0], [0, 0]];
  for (const [index, value] of truth.entries()) matrix[value][Number(scores[index] >= .5)]++;
  const [[, fp], [fn, tp]] = matrix;
  return { name, matrix, precision: tp + fp ? tp / (tp + fp) : null, recall: tp / (tp + fn) };
}

export function train(data) {
  const { train: training, val, test } = splits(data.y);
  const columns = fitColumns(data, training);
  const width = columns.reduce((total, column) => total + (column.categories?.length ?? 1), 0);
  if (width > 320) throw new Error('This CSV has too many distinct inputs for browser training. Remove unused columns or merge rare categories.');
  const X = data.X.map(row => encode(row, columns));
  const trainingRows = training.map(index => X[index]);
  const trainingLabels = training.map(index => data.y[index]);
  const estimators = [['Logistic regression', logistic(trainingRows, trainingLabels)], ['Boosted trees', boost(trainingRows, trainingLabels)]];
  const validationLabels = val.map(index => data.y[index]);
  const scored = estimators.map(([name, estimator]) => ({
    name, estimator, ap: averagePrecision(validationLabels, val.map(index => probability(estimator, X[index]))),
    scores: test.map(index => probability(estimator, X[index])),
  }));
  const winner = scored.reduce((best, model) => model.ap > best.ap ? model : best);
  const metrics = scored.map(model => evaluate(model.name, test.map(index => data.y[index]), model.scores));
  return { columns, estimator: winner.estimator, winner: winner.name, test, metrics };
}

export function ranking(data, model, indices = Array.from(data.X.keys())) {
  return indices.map(index => ({
    index, id: data.ids[index], risk: score(model, data.X[index]), actual: data.y?.[index] ?? null,
  })).sort((a, b) => b.risk - a.risk);
}

export function influences(data, model, index) {
  const row = data.X[index];
  const current = score(model, row);
  return model.columns.map((column, i) => {
    const reference = column.categories ? column.reference : column.median;
    const changed = [...row];
    changed[i] = reference;
    return { name: column.name, value: row[i] ?? 'Missing', reference, difference: current - score(model, changed) };
  }).filter(item => Math.abs(item.difference) >= .00005)
    .sort((a, b) => Math.abs(b.difference) - Math.abs(a.difference)).slice(0, 6);
}

export function groups(data, name) {
  const index = data.schema.findIndex(field => field.name === name);
  const values = data.X.map(row => row[index]);
  const amounts = [];
  if (data.schema[index].numeric) {
    const sorted = values.filter(value => value !== null).sort((a, b) => a - b);
    amounts.push(...new Set([.25, .5, .75].map(q => quantile(sorted, q)).filter(edge => edge > sorted[0] && edge < sorted.at(-1))));
    if (!amounts.length && sorted[0] < sorted.at(-1)) amounts.push(sorted[0]);
  }
  const format = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
  const counts = new Map();
  for (const [i, value] of values.entries()) {
    let group = value ?? 'Missing';
    if (value !== null && amounts.length) {
      const at = amounts.findIndex(edge => value <= edge);
      group = at === -1 ? `> ${format.format(amounts.at(-1))}` : at === 0
        ? `≤ ${format.format(amounts[0])}` : `> ${format.format(amounts[at - 1])} to ${format.format(amounts[at])}`;
    } else if (value !== null && data.schema[index].numeric && !amounts.length) {
      group = 'All recorded values';
    }
    if (!counts.has(group)) counts.set(group, { name: String(group), total: 0, left: 0 });
    const row = counts.get(group);
    row.total++;
    row.left += data.y[i];
  }
  const ordered = [...counts.values()];
  if (data.schema[index].numeric) {
    const order = amounts.map((edge, i) => i ? `> ${format.format(amounts[i - 1])} to ${format.format(edge)}` : `≤ ${format.format(edge)}`);
    if (amounts.length) order.push(`> ${format.format(amounts.at(-1))}`);
    order.push('All recorded values', 'Missing');
    return ordered.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
  }
  ordered.sort((a, b) => b.total - a.total || a.name.localeCompare(b.name));
  if (ordered.length <= 8) return ordered;
  const other = ordered.slice(7).reduce((sum, row) => ({ name: 'Other groups', total: sum.total + row.total, left: sum.left + row.left }), { total: 0, left: 0 });
  return [...ordered.slice(0, 7), other];
}

export function estimate(ranked, share, cost, value, success, budget = null, prefix = null) {
  const contacted = Math.ceil(ranked.length * share / 100);
  const expected = prefix ? prefix[contacted] : ranked.slice(0, contacted).reduce((sum, row) => sum + row.risk, 0);
  const benefit = expected * value * success / 100;
  const spend = contacted * cost;
  const net = benefit - spend;
  const maxBenefit = expected * value;
  const breakEven = maxBenefit > 0 ? 100 * spend / maxBenefit : spend > 0 ? Infinity : null;
  const overBudget = budget !== null && spend - budget > 1e-8;
  return { contacted, benefit, spend, net, breakEven, overBudget };
}
