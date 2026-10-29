const $ = (id) => document.getElementById(id);
const storageKey = 'joi.conversation.v1';
let messages = [];
let pending = false;
try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || '[]');
  if (Array.isArray(saved) && saved.every((m, i) => m && m.role === (i % 2 ? 'assistant' : 'user') && typeof m.content === 'string') && saved.length % 2 === 0) messages = saved.slice(-38);
} catch { /* A fresh conversation also works when browser storage is unavailable. */ }

function save() { try { localStorage.setItem(storageKey, JSON.stringify(messages)); } catch {} }
function scrollToEnd() { $('conversation').scrollTop = $('conversation').scrollHeight; }
function render() {
  $('messages').replaceChildren();
  $('welcome').hidden = messages.length > 0;
  $('suggestions').hidden = messages.length > 0;
  for (const message of messages) {
    const row = document.createElement('div');
    row.className = `message ${message.role}`;
    if (message.role === 'assistant') {
      const avatar = document.createElement('img');
      avatar.src = '/assets/joi-portrait.png'; avatar.alt = '';
      row.append(avatar);
    }
    const body = document.createElement('div'); body.className = 'message-content';
    const name = document.createElement('span'); name.className = 'name'; name.textContent = message.role === 'user' ? 'YOU' : 'JOI';
    const text = document.createElement('p'); text.textContent = message.content;
    body.append(name, text); row.append(body); $('messages').append(row);
  }
  scrollToEnd();
}
function setPending(value) {
  pending = value;
  $('thinking').hidden = !value;
  $('send').disabled = value;
  $('new-chat').disabled = value;
  $('message').disabled = value;
  document.querySelectorAll('[data-prompt]').forEach((button) => button.disabled = value);
  scrollToEnd();
}
function updateCounter() { $('counter').textContent = `${$('message').value.length} / 240`; }

async function send(event) {
  event?.preventDefault();
  const text = $('message').value.trim();
  if (!text || pending) return;
  $('error').hidden = true;
  messages = messages.slice(-38);
  messages.push({ role: 'user', content: text });
  $('message').value = ''; updateCounter(); render(); setPending(true);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages }), signal: AbortSignal.timeout(60000)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Joi couldn't finish that reply.");
    if (typeof result.reply !== 'string' || !result.reply.trim()) throw new Error('No reply came through. Try again?');
    messages.push({ role: 'assistant', content: result.reply });
    save(); render();
  } catch (error) {
    messages.pop(); render(); save();
    $('message').value = text; updateCounter();
    $('error').textContent = error.name === 'TimeoutError' ? 'That reply is taking a while. Your message is ready to try again.' : error.message;
    $('error').hidden = false;
  } finally { setPending(false); $('message').focus(); }
}

$('chat-form').addEventListener('submit', send);
$('message').addEventListener('input', updateCounter);
$('message').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
});
document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => { $('message').value = button.dataset.prompt; send(); }));
$('new-chat').addEventListener('click', () => { messages = []; save(); render(); $('error').hidden = true; $('message').value = ''; updateCounter(); $('message').focus(); });
$('about-open').addEventListener('click', () => $('about').showModal());
$('about-close').addEventListener('click', () => $('about').close());
$('about').addEventListener('click', (event) => { if (event.target === $('about')) $('about').close(); });

fetch('/api/status').then((response) => {
  if (!response.ok) throw new Error('Model unavailable');
  return response.json();
}).then((status) => {
  $('connection-status').textContent = 'local companion · still learning';
  $('model-note').textContent = `Our own dialogue model · ${status.parameters.toLocaleString()} learned parameters · training step ${status.step.toLocaleString()} · ${status.context_size} ${status.tokenizer === 'word' ? 'tokens' : 'characters'} of context.`;
}).catch(() => { $('connection-status').textContent = 'waiting for the local model'; $('model-note').textContent = 'The local model is not connected yet.'; });
render();
