'use strict';
const $ = (id) => document.getElementById(id);
const state = {tasks: [], task: null, sources: [], jobs: [], selected: null, rendered: new Set(), timer: null};
const terminal = new Set(['completed', 'failed', 'partial', 'cancelled', 'interrupted']);
function el(tag, cls, text) { const node = document.createElement(tag); if (cls) node.className = cls; if (text != null) node.textContent = text; return node; }
function toast(message, error = false) { $('toast').textContent = message; $('toast').className = error ? 'error' : ''; $('toast').hidden = false; clearTimeout(state.timer); state.timer = setTimeout(() => { $('toast').hidden = true; }, 8000); }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) { let body; try { body = await response.json(); } catch { body = {}; } const detail = body.detail; throw new Error(Array.isArray(detail) ? detail.map(item => item.msg).join('; ') : detail || `Request failed (${response.status})`); }
  return response.json();
}
const json = (method, body) => ({method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
function taskLabel(id) { return state.tasks.find(task => task.id === id)?.title || id; }
function selectTask(id, reset = true) {
  state.task = state.tasks.find(task => task.id === id);
  const task = state.task;
  document.querySelectorAll('.task-button').forEach(button => { button.classList.toggle('active', button.dataset.task === id); button.setAttribute('aria-current', button.dataset.task === id ? 'page' : 'false'); });
  $('task-title').textContent = task.title; $('task-description').textContent = task.description;
  $('task-group').textContent = `${task.group.toUpperCase()} / ${String(state.tasks.indexOf(task) + 1).padStart(2, '0')}`;
  $('source-panel').hidden = id === 'voice_design'; $('text-field').hidden = task.route !== 'speech';
  $('instruction-number').textContent = id === 'voice_design' ? '01' : '02';
  $('settings-number').textContent = id === 'voice_design' ? '02' : '03';
  $('instruction-field').hidden = id === 'voice_clone'; $('speed-field').hidden = id !== 'speed';
  $('instruction-label').textContent = id === 'voice_design' ? 'Voice description' : 'Editing instruction';
  $('instruction-hint').textContent = id === 'voice_clone' ? 'The reference determines the voice. For cloning with additional delivery instructions, use Free-form instruction.' : 'Edit the example to match your recording. English and Chinese instructions are supported.';
  $('duration-hint').textContent = task.route === 'speech' ? 'Explicit duration is required. Each additional take increments the seed.' : id === 'speed' ? 'Blank duration uses source length ÷ speed factor. Output must remain within 30 seconds.' : 'Blank duration matches the source. Text-only requests need a duration; length-changing edits may need an override.';
  if (reset) { $('instruction').value = task.instruction; $('speech-text').value = task.text || ''; $('duration').value = task.duration ?? ''; $('speed').value = '1.25'; }
}
function request() {
  const models = ['flash', 'base'].filter(model => $(`model-${model}`).checked);
  return {task: state.task.id, models, source_id: state.task.id === 'voice_design' ? null : $('source-select').value || null,
    text: $('speech-text').value, instruction: $('instruction').value, duration: $('duration').value === '' ? null : Number($('duration').value),
    speed_factor: Number($('speed').value), seed: Number($('seed').value), repeats: Number($('repeats').value)};
}
function renderSources(selected = $('source-select').value) {
  $('source-select').replaceChildren(new Option('No source selected', ''));
  state.sources.forEach(source => $('source-select').add(new Option(`${source.name} · ${source.duration.toFixed(2)}s`, source.id)));
  $('source-select').value = selected; sourceChanged();
}
function sourceChanged() {
  const source = state.sources.find(item => item.id === $('source-select').value);
  $('source-preview').hidden = !source;
  if (source) { $('source-audio').src = source.audio_url; $('source-info').textContent = `${source.duration.toFixed(2)}s · ${source.sample_rate / 1000} kHz · peak ${source.peak_dbfs} dBFS · RMS ${source.rms_dbfs} dBFS`; }
  else { $('source-audio').pause(); $('source-audio').removeAttribute('src'); }
}
async function refreshStatus() {
  const status = await api('/api/status'); $('services').replaceChildren();
  for (const [model, info] of Object.entries(status.models)) {
    const line = el('div', `service${info.ready ? ' ready' : ''}`); line.append(el('span', 'service-dot'), el('span', '', model === 'flash' ? 'AuK Flash' : 'AuK Base'), el('small', '', info.ready ? 'READY' : 'OFFLINE'));
    line.title = info.error || info.models.join(', '); $('services').append(line);
  }
}
function renderHistory() {
  $('history').replaceChildren();
  if (!state.jobs.length) { $('history').append(el('p', 'hint', 'No experiments yet.')); return; }
  state.jobs.forEach(job => {
    const button = el('button', 'history-row' + (job.id === state.selected ? ' selected' : '')); button.type = 'button';
    const copy = el('div'); copy.append(el('strong', '', taskLabel(job.request.task)), el('small', '', `${new Date(job.created_at).toLocaleString()} · ${job.request.models.join(' + ')} · seed ${job.request.seed}`));
    button.append(copy, el('span', 'status', job.status)); button.onclick = () => showJob(job, true); $('history').append(button);
  });
}
function metric(value, label) { const node = el('div', 'metric'); node.append(el('strong', '', value), el('small', '', label)); return node; }
function renderResult(job, result) {
  const card = el('article', 'result-card'); const top = el('div', 'result-top');
  top.append(el('h3', '', result.model === 'flash' ? 'AuK Flash' : 'AuK Base'), el('span', 'pill', `seed ${result.seed}`));
  if (result.audio_url) { const link = el('a', '', 'WAV ↓'); link.href = result.audio_url; link.download = `${job.request.task}-${result.model}-${result.seed}.wav`; top.append(link); }
  card.append(top);
  if (result.status === 'failed') { card.append(el('p', 'error', result.error)); return card; }
  const audio = el('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = result.audio_url; card.append(audio);
  const reuse = el('button', 'text-button reuse', 'Use this take as a source ↗');
  reuse.onclick = async () => { try { const source = await api(`/api/jobs/${job.id}/results/${result.id}/source`, {method: 'POST'}); state.sources.unshift(source); selectTask('custom'); renderSources(source.id); $('instruction').value = ''; $('duration').value = ''; toast('Take added as source. Choose an edit or write your own instruction.'); } catch (error) { toast(error.message, true); } };
  card.append(reuse);
  const metrics = el('div', 'metrics'); metrics.append(metric(`${result.latency_seconds.toFixed(2)}s`, 'Request time'), metric(`${result.rtf.toFixed(3)}×`, 'RTF'), metric(`${result.duration.toFixed(2)}s`, 'Output duration')); card.append(metrics);
  card.append(el('div', 'audio-info', `Δ duration ${result.duration_error_ms} ms · peak ${result.peak_dbfs} dBFS\nRMS ${result.rms_dbfs} dBFS · near/full-scale samples ${result.clipped_percent}%`));
  const details = el('details'); details.append(el('summary', '', 'Evaluate this take'));
  const grid = el('div', 'review-grid'); const selects = {};
  for (const [key, label] of [['content', 'Content'], ['identity', 'Identity'], ['instruction', 'Instruction'], ['quality', 'Quality']]) {
    const field = el('label', '', label); const select = el('select'); select.add(new Option('Unrated', '')); for (let score = 1; score <= 5; score++) select.add(new Option(`${score} / 5`, score)); select.value = result.review[key] ?? ''; selects[key] = select; field.append(select); grid.append(field);
  }
  details.append(el('p', 'hint', '1 = poor, 5 = excellent. Leave identity unrated for tasks that intentionally change the voice.'), grid);
  const notes = el('textarea', 'review-notes'); notes.placeholder = 'Pronunciation, artifacts, identity drift, useful settings…'; notes.setAttribute('aria-label', 'Listening notes'); notes.value = result.review.notes || '';
  const actions = el('div', 'review-actions'); const verdict = el('select'); verdict.setAttribute('aria-label', 'Verdict'); for (const [value, title] of [['unrated', 'No verdict'], ['keep', 'Keep / useful'], ['mixed', 'Mixed result'], ['reject', 'Reject']]) verdict.add(new Option(title, value)); verdict.value = result.review.verdict;
  const save = el('button', 'secondary', 'Save notes'); const feedback = el('span', 'review-status');
  save.onclick = async () => { save.disabled = true; try { const body = {verdict: verdict.value, notes: notes.value}; for (const key of Object.keys(selects)) body[key] = selects[key].value ? Number(selects[key].value) : null; await api(`/api/jobs/${job.id}/results/${result.id}/review`, json('PUT', body)); feedback.textContent = 'Saved'; } catch (error) { toast(error.message, true); } finally { save.disabled = false; } };
  actions.append(verdict, save); details.append(notes, actions, feedback); card.append(details);
  const provenance = el('details'); provenance.append(el('summary', '', 'Inference record'), el('pre', '', JSON.stringify({route: result.route, model: result.model_id, payload: result.payload, backend: result.backend_meta, runtime: result.runtime}, null, 2))); card.append(provenance);
  return card;
}
function showJob(job, reset = false) {
  if (reset || state.selected !== job.id) {
    state.selected = job.id; state.rendered.clear(); $('result-cards').replaceChildren();
    const source = state.sources.find(item => item.id === job.request.source_id);
    $('job-source-audio').hidden = !source; $('job-source-audio').pause();
    if (source) $('job-source-audio').src = source.audio_url; else $('job-source-audio').removeAttribute('src');
  }
  $('active-job').hidden = false; $('job-progress').textContent = `${taskLabel(job.request.task)} · ${job.status}\n${job.error || job.progress}`;
  $('cancel-job').hidden = terminal.has(job.status); $('cancel-job').disabled = job.status === 'cancelling';
  for (const result of job.results) { if (!state.rendered.has(result.id)) { $('result-cards').append(renderResult(job, result)); state.rendered.add(result.id); } }
  if (!job.results.length && terminal.has(job.status) && !state.rendered.has('empty')) { $('result-cards').append(el('p', 'hint', job.error || 'No completed audio in this experiment.')); state.rendered.add('empty'); }
  renderHistory();
}
async function refreshJobs() {
  state.jobs = await api('/api/jobs');
  if (state.selected) { const selected = state.jobs.find(job => job.id === state.selected); if (selected) showJob(selected); }
  else renderHistory();
}
async function upload() {
  if (!$('file').files[0]) return toast('Choose a file first.', true);
  const data = new FormData(); data.append('file', $('file').files[0]); data.append('start', $('crop-start').value || '0'); if ($('crop-end').value) data.append('end', $('crop-end').value);
  $('upload-button').disabled = true; $('upload-button').textContent = 'Preparing…';
  try { const source = await api('/api/sources', {method: 'POST', body: data}); state.sources.unshift(source); renderSources(source.id); toast('Clip prepared. Original silence and volume preserved.'); }
  catch (error) { toast(error.message, true); } finally { $('upload-button').disabled = false; $('upload-button').textContent = 'Prepare clip'; }
}
async function init() {
  const catalog = await api('/api/catalog'); state.tasks = catalog.tasks; $('task-count').textContent = state.tasks.length;
  let group;
  state.tasks.forEach(task => { if (group !== task.group) { group = task.group; $('tasks').append(el('div', 'nav-group', group)); } const button = el('button', 'task-button', task.title); button.type = 'button'; button.dataset.task = task.id; button.onclick = () => selectTask(task.id); $('tasks').append(button); });
  selectTask('voice_design');
  state.sources = await api('/api/sources'); renderSources(); await refreshJobs();
  if (state.jobs.length) showJob(state.jobs[0], true);
  await refreshStatus();
  // Schedule after completion to avoid overlapping requests on a slow or sleeping server.
  async function pollJobs() { try { await refreshJobs(); } catch (error) { console.warn(error); } setTimeout(pollJobs, 1800); }
  async function pollStatus() { try { await refreshStatus(); } catch (error) { console.warn(error); } setTimeout(pollStatus, 15000); }
  setTimeout(pollJobs, 1800); setTimeout(pollStatus, 15000);
}
$('upload-button').onclick = upload; $('source-select').onchange = sourceChanged;
$('clear-source').onclick = () => { $('source-select').value = ''; sourceChanged(); };
$('reset-template').onclick = () => selectTask(state.task.id);
$('speed').onchange = () => { $('instruction').value = `Adjust the speech speed to ${$('speed').value}x.`; };
$('refresh-status').onclick = () => refreshStatus().catch(error => toast(error.message, true));
$('preview-request').onclick = async () => { try { $('request-preview').textContent = JSON.stringify(await api('/api/preview', json('POST', request())), null, 2); } catch (error) { toast(error.message, true); } };
$('run-form').onsubmit = async (event) => { event.preventDefault(); $('run-button').disabled = true; try { const job = await api('/api/jobs', json('POST', request())); state.jobs.unshift(job); showJob(job, true); toast('Experiment queued. You can keep exploring while it runs.'); } catch (error) { toast(error.message, true); } finally { $('run-button').disabled = false; } };
$('cancel-job').onclick = async () => { try { const job = await api(`/api/jobs/${state.selected}/cancel`, {method: 'POST'}); showJob(job); } catch (error) { toast(error.message, true); } };
$('load-settings').onclick = () => {
  const job = state.jobs.find(item => item.id === state.selected); if (!job) return;
  const saved = job.request; selectTask(saved.task, false);
  $('speech-text').value = saved.text; $('instruction').value = saved.instruction; $('duration').value = saved.duration ?? '';
  $('seed').value = saved.seed; $('repeats').value = saved.repeats; $('speed').value = saved.speed_factor;
  ['flash', 'base'].forEach(model => { $(`model-${model}`).checked = saved.models.includes(model); });
  $('source-select').value = saved.source_id || ''; sourceChanged(); toast('Experiment settings restored.');
};
document.addEventListener('play', (event) => { if (event.target instanceof HTMLAudioElement) document.querySelectorAll('audio').forEach(audio => { if (audio !== event.target) audio.pause(); }); }, true);
init().catch(error => toast(`Could not initialize AuK Lab: ${error.message}`, true));
