import { estimate } from './engine.mjs';

const $ = id => document.getElementById(id);
const num = new Intl.NumberFormat('en');
const percent = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 1 });
const decimal = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
const money = new Intl.NumberFormat('en', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
const pageSize = 8;
const riskRanges = ['Under 25%', '25% to under 50%', '50% to under 75%', '75% to 100%'];
let data = null, selected = null, activeRange = null, page = 0, prefix = [];
let view = 'risk', busy = false, plot = null;
let sequence = 0, detailSequence = 0, groupSequence = 0;
const requests = new Map();
const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
worker.onmessage = ({ data: { id, result, error } }) => {
  const request = requests.get(id);
  requests.delete(id);
  if (error) request?.reject(new Error(error));
  else request?.resolve(result);
};
worker.onerror = () => {
  for (const { reject } of requests.values()) reject(new Error('The analysis could not start. Reload the page and try again.'));
  requests.clear();
};

function ask(action, options = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    requests.set(id, { resolve, reject });
    worker.postMessage({ id, action, ...options });
  });
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

const displayNames = {
  tenure: 'Tenure (months)', MonthlyCharges: 'Monthly charges', TotalCharges: 'Total charges',
  SeniorCitizen: 'Senior citizen', account_length: 'Account length', number_vmail_messages: 'Voicemail messages',
  number_customer_service_calls: 'Customer service calls', area_code: 'Area code',
  international_plan: 'International plan', voice_mail_plan: 'Voicemail plan',
};
function fieldName(name) {
  if (Object.hasOwn(displayNames, name)) return displayNames[name];
  const parts = name.match(/^total_(day|eve|night|intl)_(minutes|calls|charge)$/);
  if (parts) {
    const period = { day: 'Daytime', eve: 'Evening', night: 'Night', intl: 'International' }[parts[1]];
    return `${period} ${parts[2] === 'charge' ? 'charges' : parts[2]}`;
  }
  const label = name.replace(/([a-z])([A-Z])/g, '$1 $2').replaceAll('_', ' ');
  return label[0].toUpperCase() + label.slice(1);
}

function displayValue(value) {
  if (typeof value === 'number') return decimal.format(value);
  if (value.startsWith('area_code_')) return value.slice(10);
  return value.length <= 2 ? value.toUpperCase() : value[0]?.toUpperCase() + value.slice(1);
}

function bar(label, text, width, options = {}) {
  const row = element(options.onClick ? 'button' : 'div', undefined, 'bar-item');
  if (options.onClick) {
    row.type = 'button';
    row.setAttribute('aria-pressed', String(options.pressed));
    row.addEventListener('click', options.onClick);
  }
  const caption = element('span', undefined, 'bar-caption');
  const name = element('span', label, 'bar-label');
  const value = element('span', text, 'bar-value');
  if (options.note) value.append(element('small', ` · ${options.note}`));
  caption.append(name, value);
  const track = element('span', undefined, 'track');
  track.setAttribute('aria-hidden', 'true');
  const fill = element('span', undefined, `fill ${options.color ?? ''}`);
  fill.style.width = `${Math.min(100, Math.max(0, width))}%`;
  track.append(fill);
  row.append(caption, track);
  return row;
}

function setBusy(loading) {
  busy = loading;
  for (const id of ['browse', 'demo', 'another', 'reset', 'confirm-label']) $(id).disabled = loading;
  $('feedback').setAttribute('aria-busy', String(loading));
}

function feedbackPosition() {
  if (data) document.querySelector('.dashboard-head').after($('feedback'));
  else $('intake').append($('feedback'));
}

function notice(message, error = false) {
  feedbackPosition();
  const box = $('notice');
  box.hidden = !message;
  box.textContent = message ?? '';
  box.classList.toggle('error', error);
  if (error) {
    box.tabIndex = -1;
    box.focus();
  }
}

function hideMapping() { $('mapping').hidden = true; }

async function load(action, options = {}) {
  if (busy) return;
  hideMapping();
  setBusy(true);
  notice(action === 'demo' ? 'Opening the IBM Telco demo…' : 'Preparing your analysis…');
  try {
    const result = await ask(action, options);
    if (result.pending) {
      notice('');
      fillSelect($('label-choice'), [
        { value: '', text: result.canScore ? 'Score customers without known outcomes' : 'Choose an outcome column' },
        ...result.options.map(value => ({ value, text: value })),
      ], result.options.length === 1 && !result.canScore ? result.options[0] : '');
      $('mapping').hidden = false;
      $('label-choice').focus();
      return;
    }
    showDataset(result);
    notice('');
  } catch (error) {
    notice(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function openFile(file) {
  if (!file || busy) return;
  hideMapping();
  if (!file.size || file.size > 10 * 1024 * 1024) {
    notice('Choose a nonempty CSV no larger than 10 MB.', true);
    return;
  }
  if (!/\.csv$/i.test(file.name)) {
    notice('Choose a file with the .csv extension.', true);
    return;
  }
  setBusy(true);
  try {
    const buffer = await file.arrayBuffer();
    const text = new TextDecoder('utf-8', { fatal: true }).decode(buffer);
    setBusy(false);
    await load('inspect', { text, name: file.name });
  } catch {
    notice('The file could not be read. Export it as a UTF-8 CSV and try again.', true);
  } finally {
    setBusy(false);
  }
}

function fillSelect(select, choices, value) {
  select.replaceChildren(...choices.map(option => {
    const result = element('option', option.text);
    result.value = option.value;
    return result;
  }));
  select.value = value;
}

function showDataset(result) {
  data = result;
  page = 0;
  activeRange = null;
  selected = data.ranked[0];
  $('search').value = '';
  $('landing').hidden = true;
  $('dashboard').hidden = false;
  $('tab-history').hidden = !data.history;
  const modes = {
    demo: ['Demo data', 'Held-out scores from the public IBM Telco dataset'],
    historical: ['Historical analysis', 'Estimated risk on held-out customers'],
    'active-trained': ['New customer predictions', 'Uses the model trained on your historical CSV'],
    'active-ibm': ['New customer predictions', 'Uses the built-in IBM Telco model'],
    'active-2020': ['New customer predictions', 'Uses the built-in Churn Prediction 2020 model'],
  };
  const [type, description] = modes[data.mode];
  $('dataset-type').textContent = type;
  $('dataset-name').textContent = data.name;
  $('dataset-meta').textContent = `${num.format(data.ranked.length)} customers scored · ${description}.`;
  $('notes').replaceChildren(...data.notes.map(note => element('li', note)));
  $('data-notes').hidden = !data.notes.length;
  $('data-notes').open = false;
  prefix = [0];
  for (const row of data.ranked) prefix.push(prefix.at(-1) + row.risk);
  if (data.history) renderHistory();
  renderBins();
  renderList();
  renderCustomer(selected);
  setView('risk');
  $('dataset-name').focus();
}

function setView(name, focus = false) {
  view = name;
  for (const option of ['risk', 'plan', 'history']) {
    const active = option === name;
    const tab = $(`tab-${option}`);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    $(option).hidden = !active;
    if (active && focus) tab.focus();
  }
  if (name === 'plan') renderPlan();
}

function binFor(risk) { return Math.min(3, Math.floor(risk * 4)); }

function renderBins() {
  const counts = [0, 0, 0, 0];
  for (const row of data.ranked) counts[binFor(row.risk)]++;
  $('risk-bins').replaceChildren(...riskRanges.map((label, i) => {
    const share = counts[i] / data.ranked.length;
    return bar(label, num.format(counts[i]), share * 100, {
      note: percent.format(share),
      pressed: activeRange === i,
      onClick: () => {
        activeRange = activeRange === i ? null : i;
        page = 0;
        renderBins();
        renderList();
        $('risk-bins').children[i].focus();
      },
    });
  }));
}

function renderList() {
  const query = $('search').value.trim().toLowerCase();
  const filtered = data.ranked.filter(row => (activeRange === null || binFor(row.risk) === activeRange) && row.id.toLowerCase().includes(query));
  const start = page * pageSize;
  const rows = filtered.slice(start, start + pageSize);
  const body = $('rank-body');
  if (!rows.length) {
    const cell = element('td', 'No customers match the current filters.', 'empty');
    cell.colSpan = 2;
    const row = element('tr');
    row.append(cell);
    body.replaceChildren(row);
  } else {
    body.replaceChildren(...rows.map(record => {
      const row = element('tr');
      const name = element('td');
      const profile = element('button', record.id);
      profile.type = 'button';
      profile.className = 'customer-link';
      profile.setAttribute('aria-label', `View customer ${record.id}`);
      if (record.index === selected?.index) profile.setAttribute('aria-current', 'true');
      profile.addEventListener('click', () => {
        selected = record;
        for (const link of body.querySelectorAll('.customer-link')) link.removeAttribute('aria-current');
        profile.setAttribute('aria-current', 'true');
        renderCustomer(record);
        $('customer-card').focus({ preventScroll: true });
        const visible = $('customer-card').getBoundingClientRect();
        if (visible.top < 0 || visible.bottom > window.innerHeight) $('customer-card').scrollIntoView({ block: 'start' });
      });
      name.append(profile);
      row.append(name, element('td', percent.format(record.risk)));
      return row;
    }));
  }
  $('range-count').textContent = rows.length ? `${num.format(start + 1)}–${num.format(start + rows.length)} of ${num.format(filtered.length)}` : '0 customers';
  $('previous').disabled = page === 0;
  $('next').disabled = start + pageSize >= filtered.length;
  $('clear-filter').hidden = activeRange === null;
  $('clear-filter').textContent = activeRange === null ? '' : `Clear risk filter · ${riskRanges[activeRange]}`;
}

async function renderCustomer(record) {
  const token = ++detailSequence;
  $('customer-id').textContent = record.id;
  $('customer-score').textContent = percent.format(record.risk);
  $('influences').replaceChildren(element('p', 'Loading customer details…', 'fine'));
  try {
    const values = await ask('details', { index: record.index });
    if (token !== detailSequence) return;
    if (!values.length) {
      $('influences').replaceChildren(element('p', 'There were no visible changes when comparing these inputs with typical values.', 'fine'));
      return;
    }
    const scale = Math.max(...values.map(row => Math.abs(row.difference)));
    $('influences').replaceChildren(...values.map(({ name, value, difference }) => {
      const label = `${fieldName(name)} · ${displayValue(value)}`;
      const sign = difference > 0 ? '+' : '−';
      const amount = Math.abs(difference * 100);
      const text = `${sign}${amount < .1 ? '<0.1' : decimal.format(amount)} pp`;
      return bar(label, text, 100 * Math.abs(difference) / scale, { color: difference > 0 ? 'loss' : 'gain' });
    }));
  } catch {
    if (token === detailSequence) $('influences').replaceChildren(element('p', 'Customer details are temporarily unavailable.', 'fine'));
  }
}

function exportCSV() {
  const rows = data.ranked.map(({ id, risk }) => {
    const safe = /^[\s\uFEFF]*[=+@\-]/.test(id) ? `'${id}` : id;
    return `"${safe.replaceAll('"', '""')}",${risk.toFixed(8)}`;
  });
  const url = URL.createObjectURL(new Blob(['customerID,risk_score\r\n', rows.join('\r\n'), '\r\n'], { type: 'text/csv;charset=utf-8' }));
  const link = element('a');
  link.href = url;
  link.download = 'churn_rankings.csv';
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}

function renderHistory() {
  const history = data.history;
  const stayed = history.size - history.left;
  $('outcomes').replaceChildren(
    bar('Customers who stayed', num.format(stayed), 100 * stayed / history.size, { note: percent.format(stayed / history.size), color: 'slate' }),
    bar('Customers who left', num.format(history.left), 100 * history.left / history.size, { note: percent.format(history.left / history.size), color: 'loss' }),
  );
  const field = history.fields.includes('Contract') ? 'Contract' : history.fields[0];
  fillSelect($('group-field'), history.fields.map(value => ({ value, text: fieldName(value) })), field);
  renderGroups();
  $('evaluation-meta').textContent = `${num.format(history.testSize)} held-out customers. The selected model is ${data.winner}. Customers at 50% estimated risk or higher are flagged.`;
  $('model-chart').replaceChildren(...history.metrics.map(record => {
    const card = element('section', undefined, 'model-card');
    card.append(element('h3', record.name));
    for (const [label, value] of [['Flagged customers who left', record.precision], ['Actual churners found', record.recall]]) {
      card.append(bar(label, value === null ? 'No customers flagged' : percent.format(value), 100 * (value ?? 0)));
    }
    return card;
  }));
  fillSelect($('evaluation-model'), history.metrics.map(({ name }) => ({ value: name, text: name })), data.winner);
  renderEvaluationCounts();
}

async function renderGroups() {
  const token = ++groupSequence;
  try {
    const groups = await ask('group', { field: $('group-field').value });
    if (token !== groupSequence) return;
    $('groups').replaceChildren(...groups.map(({ name, left, total }) => (
      bar(displayValue(name), percent.format(left / total), 100 * left / total, { note: `${num.format(left)} of ${num.format(total)}` })
    )));
  } catch {
    if (token === groupSequence) $('groups').replaceChildren(element('p', 'Group results are temporarily unavailable.', 'fine'));
  }
}

function renderEvaluationCounts() {
  const record = data.history.metrics.find(({ name }) => name === $('evaluation-model').value);
  const [[tn, fp], [fn, tp]] = record.matrix;
  $('evaluation-counts').replaceChildren(...[
    ['Churners flagged', tp], ['Churners missed', fn], ['False alarms', fp], ['Stayers correctly unflagged', tn],
  ].map(([label, value]) => {
    const node = element('div');
    node.append(element('strong', num.format(value)), document.createTextNode(label));
    return node;
  }));
}

function renderPlan() {
  if (view !== 'plan' || !data) return;
  const share = $('share').valueAsNumber, success = $('success').valueAsNumber;
  const cost = $('cost').valueAsNumber, value = $('value').valueAsNumber;
  const budget = $('budget').value === '' ? null : $('budget').valueAsNumber;
  $('share-value').textContent = `${share}%`;
  $('success-value').textContent = `${success}%`;
  const valid = ['cost', 'value', 'budget'].every(id => $(id).validity.valid);
  $('invalid-plan').hidden = valid;
  $('plan-values').hidden = !valid;
  $('curve-card').hidden = !valid;
  if (!valid) {
    $('invalid-plan').textContent = 'Enter a valid cost and customer value between 0 and 1,000,000,000. Leave the budget blank or enter an amount in the same range.';
    plot = null;
    return;
  }
  const get = amount => estimate(data.ranked, amount, cost, value, success, budget, prefix);
  const result = get(share);
  let status = 'neutral';
  let label = 'No customers contacted';
  let text = 'Leaving the offer off costs zero and provides the baseline for comparison.';
  if (result.contacted) {
    if (result.net < -.005) {
      status = 'loss';
      label = 'Estimated loss';
      text = 'At these assumptions, offer costs exceed the avoided loss.';
    } else if (result.net > .005) {
      status = 'gain';
      label = 'Estimated gain';
      text = 'This gain depends on the assumed offer success. Actual results may differ.';
    } else {
      label = 'Estimated break-even';
      text = 'At these assumptions, estimated avoided loss and offer costs balance.';
    }
    if (result.breakEven > 100) text = 'Even 100% offer success would not cover the modeled cost.';
  }
  $('decision').dataset.status = result.overBudget ? 'budget' : status;
  $('decision-state').textContent = result.overBudget ? `Over budget · ${label.toLowerCase()}` : label;
  $('net').textContent = Math.abs(result.net) < .005 ? money.format(0) : `${result.net > 0 ? '+' : '−'}${money.format(Math.abs(result.net))}`;
  $('decision-copy').textContent = text;
  $('contacted').textContent = num.format(result.contacted);
  $('break-even').textContent = !result.contacted || result.breakEven === null ? '—'
    : result.breakEven > 100 ? 'More than 100%' : percent.format(result.breakEven / 100);
  const maximum = Math.max(result.spend, result.benefit) || 1;
  $('plan-bars').replaceChildren(
    bar('Estimated avoided loss', money.format(result.benefit), 100 * result.benefit / maximum, { color: 'gain' }),
    bar('Total offer cost', money.format(result.spend), 100 * result.spend / maximum, { color: 'loss' }),
  );
  $('budget-result').hidden = budget === null;
  $('budget-result').textContent = budget === null ? ''
    : result.overBudget ? `Over budget by ${money.format(result.spend - budget)}.` : `Within budget · ${money.format(Math.max(0, budget - result.spend))} remaining.`;
  const series = Array.from({ length: 101 }, (_, i) => get(i).net);
  plot = { series, share, status, geometry: null };
  $('curve-note').textContent = `Selected share: ${share}% · ${num.format(result.contacted)} customers · ${label.toLowerCase() === 'no customers contacted' ? 'net savings' : label.toLowerCase()} ${$('net').textContent}.`;
  drawCurve();
}

function drawCurve() {
  if (!plot || view !== 'plan') return;
  const canvas = $('curve');
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  const ctx = canvas.getContext('2d');
  ctx.scale(pixelRatio, pixelRatio);
  const { share, series, status } = plot;
  const maximum = Math.max(0, ...series), minimum = Math.min(0, ...series);
  const span = maximum - minimum || 1;
  const upper = maximum + span * .12, lower = minimum - span * .12;
  const left = 55, right = width - 15, top = 22, bottom = height - 37;
  const x = i => left + (right - left) * i / 100;
  const y = value => top + (bottom - top) * (upper - value) / (upper - lower);
  plot.geometry = { left, right };
  ctx.font = '12px system-ui';
  ctx.fillStyle = '#53627c';
  ctx.textAlign = 'left';
  ctx.fillText('Net savings', 1, 12);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const ticks = [minimum, 0, maximum].filter((value, i, all) => all.indexOf(value) === i);
  let lastLabelY = null;
  for (const value of ticks.sort((a, b) => Math.abs(a) - Math.abs(b))) {
    const at = y(value);
    ctx.beginPath();
    ctx.setLineDash(value === 0 ? [4, 4] : []);
    ctx.strokeStyle = value === 0 ? '#7786a2' : '#e1e8f5';
    ctx.moveTo(left, at);
    ctx.lineTo(right, at);
    ctx.stroke();
    if (lastLabelY === null || Math.abs(at - lastLabelY) > 16) {
      ctx.fillStyle = '#53627c';
      ctx.fillText(compact.format(value), left - 7, at);
      if (value === 0) lastLabelY = at;
    }
  }
  ctx.setLineDash([]);
  ctx.fillStyle = '#53627c';
  ctx.textAlign = 'center';
  for (const value of [0, 25, 50, 75, 100]) {
    ctx.fillText(`${value}%`, Math.min(right - 3, x(value)), height - 18);
  }
  ctx.beginPath();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#4162da';
  for (const [i, value] of series.entries()) {
    if (i === 0) ctx.moveTo(x(i), y(value));
    else ctx.lineTo(x(i), y(value));
  }
  ctx.stroke();
  const px = x(share), py = y(series[share]);
  ctx.beginPath();
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#8394b5';
  ctx.moveTo(px, top);
  ctx.lineTo(px, bottom);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(px, py, 5, 0, Math.PI * 2);
  ctx.fillStyle = status === 'loss' ? '#a53b32' : status === 'gain' ? '#0b715f' : '#3056d8';
  ctx.fill();
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

$('browse').addEventListener('click', () => $('file').click());
$('another').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', ({ target }) => {
  const file = target.files[0];
  target.value = '';
  openFile(file);
});
$('demo').addEventListener('click', () => load('demo'));
$('confirm-label').addEventListener('click', () => load('label', { label: $('label-choice').value }));
$('reset').addEventListener('click', async () => {
  if (busy) return;
  setBusy(true);
  ++detailSequence;
  ++groupSequence;
  try {
    await ask('reset');
    data = selected = plot = null;
    $('assumptions').reset();
    $('dashboard').hidden = true;
    $('landing').hidden = false;
    hideMapping();
    notice('');
    setBusy(false);
    $('browse').focus();
  } catch (error) { notice(error.message, true); }
  finally { setBusy(false); }
});
for (const event of ['dragenter', 'dragover', 'dragleave', 'drop']) {
  $('dropzone').addEventListener(event, e => {
    e.preventDefault();
    $('dropzone').classList.toggle('drag', event === 'dragenter' || event === 'dragover');
    if (event === 'drop') {
      if (e.dataTransfer.files.length !== 1) notice('Drop one CSV at a time.', true);
      else openFile(e.dataTransfer.files[0]);
    }
  });
}
for (const option of ['risk', 'plan', 'history']) {
  $(`tab-${option}`).addEventListener('click', () => setView(option));
  $(`tab-${option}`).addEventListener('keydown', e => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    const tabs = data.history ? ['risk', 'plan', 'history'] : ['risk', 'plan'];
    const index = tabs.indexOf(option);
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1
      : (index + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    setView(tabs[next], true);
  });
}
$('previous').addEventListener('click', () => { page--; renderList(); });
$('next').addEventListener('click', () => { page++; renderList(); });
$('search').addEventListener('input', () => { page = 0; renderList(); });
$('clear-filter').addEventListener('click', () => {
  activeRange = null;
  page = 0;
  renderBins();
  renderList();
  $('search').focus();
});
$('export').addEventListener('click', exportCSV);
$('group-field').addEventListener('change', renderGroups);
$('evaluation-model').addEventListener('change', renderEvaluationCounts);
$('assumptions').addEventListener('submit', event => event.preventDefault());
$('assumptions').addEventListener('input', renderPlan);
$('curve').addEventListener('click', event => {
  if (!plot?.geometry) return;
  const { left, right } = plot.geometry;
  const offset = event.clientX - $('curve').getBoundingClientRect().left;
  $('share').value = String(Math.max(0, Math.min(100, Math.round(100 * (offset - left) / (right - left)))));
  renderPlan();
});
new ResizeObserver(drawCurve).observe($('curve'));
