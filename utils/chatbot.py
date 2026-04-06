"""
CineBot — Fixed Floating 9:16 chat widget powered by OpenRouter (free models).
Injects FAB + panel into the parent Streamlit page via window.parent.document,
so position:fixed works correctly with zero clipping.
Get free key at: https://openrouter.ai/keys
"""

import os
import json
import ast
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()


def _safe_list(val):
    if isinstance(val, list):
        return val
    try:
        r = ast.literal_eval(str(val))
        return r if isinstance(r, list) else []
    except Exception:
        return []


def _build_movies_json(df: pd.DataFrame, n: int = 300) -> str:
    top = df.nlargest(n, "vote_average")
    rows = []
    for _, row in top.iterrows():
        gl   = _safe_list(row.get("genres_list", []))
        year = int(row["release_year"]) if pd.notna(row.get("release_year")) and row.get("release_year", 0) > 0 else 0
        rows.append({
            "t": str(row["title"]),
            "g": ", ".join(gl[:3]),
            "r": round(float(row.get("vote_average", 0)), 1),
            "y": year,
            "d": str(row.get("director", ""))[:25],
            "m": str(row.get("mood", "")),
        })
    return json.dumps(rows, ensure_ascii=False)


def render_chatbot_widget(df: pd.DataFrame) -> None:
    """
    Injects the CineBot floating panel into the parent Streamlit page DOM
    via window.parent.document — bypasses iframe clipping completely.
    """
    import streamlit.components.v1 as components

    api_key     = OPENROUTER_API_KEY
    has_key     = bool(api_key and len(api_key) > 10)
    movies_json = _build_movies_json(df)

    CSS = """
#cinebot-fab, #cinebot-panel, #cinebot-panel * {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  box-sizing: border-box !important;
}

#cinebot-fab {
  position: fixed !important;
  bottom: 28px !important;
  right: 28px !important;
  width: 60px !important;
  height: 60px !important;
  border-radius: 50% !important;
  background: linear-gradient(135deg, #e50914 0%, #8b0000 100%) !important;
  border: none !important;
  cursor: pointer !important;
  font-size: 26px !important;
  color: #fff !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: 0 4px 20px rgba(229, 9, 20, 0.5), 0 2px 8px rgba(0, 0, 0, 0.4) !important;
  z-index: 2147483646 !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
  padding: 0 !important;
  margin: 0 !important;
}

#cinebot-fab:hover {
  transform: scale(1.1) !important;
  box-shadow: 0 6px 28px rgba(229, 9, 20, 0.7), 0 4px 12px rgba(0, 0, 0, 0.5) !important;
}

#cinebot-panel {
  position: fixed !important;
  bottom: 100px !important;
  right: 28px !important;
  width: 350px !important;
  height: 622px !important;
  background: #0e0e0e !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 22px !important;
  box-shadow: 0 24px 72px rgba(0, 0, 0, 0.85), 0 4px 20px rgba(0, 0, 0, 0.5) !important;
  display: none !important;
  flex-direction: column !important;
  overflow: hidden !important;
  z-index: 2147483645 !important;
  transform-origin: bottom right !important;
  margin: 0 !important;
  padding: 0 !important;
}

#cinebot-panel.cb-open {
  display: flex !important;
  animation: cbSlideIn 0.28s cubic-bezier(0.34, 1.4, 0.64, 1) !important;
}

@keyframes cbSlideIn {
  from {
    opacity: 0;
    transform: scale(0.88) translateY(20px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

#cb-hdr {
  background: linear-gradient(135deg, #e50914 0%, #7a0000 100%) !important;
  padding: 0 16px !important;
  height: 64px !important;
  min-height: 64px !important;
  flex-shrink: 0 !important;
  display: flex !important;
  align-items: center !important;
  gap: 12px !important;
  border-radius: 22px 22px 0 0 !important;
  z-index: 1 !important;
}

#cb-avatar {
  width: 40px !important;
  height: 40px !important;
  background: rgba(255, 255, 255, 0.18) !important;
  border-radius: 50% !important;
  border: 2px solid rgba(255, 255, 255, 0.3) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 20px !important;
  flex-shrink: 0 !important;
}

#cb-info {
  flex: 1 !important;
  min-width: 0 !important;
}

#cb-name {
  color: #fff !important;
  font-size: 16px !important;
  font-weight: 700 !important;
  line-height: 1.2 !important;
  margin: 0 !important;
}

#cb-status {
  color: rgba(255, 255, 255, 0.8) !important;
  font-size: 11.5px !important;
  margin-top: 3px !important;
  display: flex !important;
  align-items: center !important;
  gap: 5px !important;
}

#cb-dot {
  width: 7px !important;
  height: 7px !important;
  background: #4eff91 !important;
  border-radius: 50% !important;
  flex-shrink: 0 !important;
  box-shadow: 0 0 6px rgba(78, 255, 145, 0.6) !important;
}

#cb-close {
  background: rgba(255, 255, 255, 0.15) !important;
  border: 1px solid rgba(255, 255, 255, 0.25) !important;
  color: #fff !important;
  cursor: pointer !important;
  width: 30px !important;
  height: 30px !important;
  border-radius: 50% !important;
  font-size: 14px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex-shrink: 0 !important;
  transition: background 0.15s !important;
  line-height: 1 !important;
  padding: 0 !important;
  margin: 0 !important;
}

#cb-close:hover {
  background: rgba(255, 255, 255, 0.3) !important;
}

#cb-msgs {
  flex: 1 !important;
  overflow-y: auto !important;
  padding: 14px 12px !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 10px !important;
  scroll-behavior: smooth !important;
}

#cb-msgs::-webkit-scrollbar {
  width: 3px !important;
}

#cb-msgs::-webkit-scrollbar-thumb {
  background: #2e2e2e !important;
  border-radius: 2px !important;
}

.cb-msg {
  display: flex !important;
  flex-direction: column !important;
  max-width: 84% !important;
}

.cb-msg.u {
  align-self: flex-end !important;
  align-items: flex-end !important;
}

.cb-msg.b {
  align-self: flex-start !important;
  align-items: flex-start !important;
}

.cb-bbl {
  padding: 10px 13px !important;
  border-radius: 18px !important;
  font-size: 12.5px !important;
  line-height: 1.55 !important;
  word-break: break-word !important;
  white-space: pre-wrap !important;
  margin: 0 !important;
}

.cb-msg.u .cb-bbl {
  background: #e50914 !important;
  color: #fff !important;
  border-bottom-right-radius: 4px !important;
}

.cb-msg.b .cb-bbl {
  background: #1c1c1c !important;
  color: #ddd !important;
  border: 1px solid #2e2e2e !important;
  border-bottom-left-radius: 4px !important;
}

.cb-msg.err .cb-bbl {
  background: #1a0a00 !important;
  color: #ffb347 !important;
  border: 1px solid #3a2000 !important;
  border-bottom-left-radius: 4px !important;
}

.cb-ts {
  font-size: 10px !important;
  color: #555 !important;
  margin-top: 3px !important;
  padding: 0 3px !important;
}

#cb-typing {
  align-self: flex-start !important;
  padding: 10px 14px !important;
  background: #1c1c1c !important;
  border: 1px solid #2e2e2e !important;
  border-radius: 18px !important;
  border-bottom-left-radius: 4px !important;
  display: none !important;
  gap: 5px !important;
  align-items: center !important;
}

#cb-typing span {
  width: 6px !important;
  height: 6px !important;
  background: #666 !important;
  border-radius: 50% !important;
  animation: cbDot 1.3s infinite !important;
}

#cb-typing span:nth-child(2) {
  animation-delay: 0.2s !important;
}

#cb-typing span:nth-child(3) {
  animation-delay: 0.4s !important;
}

@keyframes cbDot {
  0%, 60%, 100% {
    transform: translateY(0);
    background: #555;
  }
  30% {
    transform: translateY(-5px);
    background: #e50914;
  }
}

#cb-qp {
  padding: 8px 10px !important;
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 5px !important;
  flex-shrink: 0 !important;
  border-top: 1px solid #1e1e1e !important;
  background: #0e0e0e !important;
}

.cb-q {
  background: #161616 !important;
  border: 1px solid #2e2e2e !important;
  color: #999 !important;
  border-radius: 12px !important;
  padding: 4px 9px !important;
  font-size: 11px !important;
  cursor: pointer !important;
  white-space: nowrap !important;
  transition: all 0.15s !important;
  margin: 0 !important;
}

.cb-q:hover {
  background: #e50914 !important;
  color: #fff !important;
  border-color: #e50914 !important;
}

#cb-inp-area {
  padding: 10px 12px !important;
  border-top: 1px solid #1e1e1e !important;
  display: flex !important;
  gap: 8px !important;
  align-items: flex-end !important;
  flex-shrink: 0 !important;
  background: #0e0e0e !important;
}

#cb-inp {
  flex: 1 !important;
  background: #161616 !important;
  border: 1px solid #2e2e2e !important;
  border-radius: 18px !important;
  padding: 9px 14px !important;
  color: #fff !important;
  font-size: 12.5px !important;
  outline: none !important;
  resize: none !important;
  max-height: 72px !important;
  overflow-y: auto !important;
  line-height: 1.4 !important;
  font-family: inherit !important;
  transition: border-color 0.15s !important;
  margin: 0 !important;
}

#cb-inp:focus {
  border-color: #e50914 !important;
}

#cb-inp::placeholder {
  color: #555 !important;
}

#cb-snd {
  width: 36px !important;
  height: 36px !important;
  background: #e50914 !important;
  border: none !important;
  border-radius: 50% !important;
  color: #fff !important;
  font-size: 16px !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex-shrink: 0 !important;
  transition: background 0.15s !important;
  padding: 0 !important;
  margin: 0 !important;
}

#cb-snd:hover {
  background: #b00710 !important;
}

#cb-snd:disabled {
  background: #2a2a2a !important;
  cursor: not-allowed !important;
}

.cb-loading {
  display: flex !important;
  gap: 4px !important;
  align-items: center !important;
  padding: 10px 13px !important;
}

.cb-loading span {
  width: 8px !important;
  height: 8px !important;
  background: #e50914 !important;
  border-radius: 50% !important;
  animation: cbLoadingDot 1.4s infinite !important;
  display: inline-block !important;
}

.cb-loading span:nth-child(1) {
  animation-delay: 0s !important;
}

.cb-loading span:nth-child(2) {
  animation-delay: 0.2s !important;
}

.cb-loading span:nth-child(3) {
  animation-delay: 0.4s !important;
}

@keyframes cbLoadingDot {
  0%, 60%, 100% {
    opacity: 0.3;
    transform: translateY(0);
  }
  30% {
    opacity: 1;
    transform: translateY(-8px);
  }
}

#cb-nokey {
  flex: 1 !important;
  display: none !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 24px 20px !important;
  text-align: center !important;
  color: #888 !important;
  font-size: 12.5px !important;
  line-height: 1.8 !important;
  gap: 4px !important;
}

#cb-nokey a {
  color: #e50914 !important;
  text-decoration: none !important;
}

#cb-nokey code {
  background: #1a1a1a !important;
  color: #ffb347 !important;
  padding: 2px 6px !important;
  border-radius: 4px !important;
  font-size: 11.5px !important;
}
"""

    PANEL_HTML = """
<button id="cinebot-fab">🎬</button>
<div id="cinebot-panel">
  <div id="cb-hdr">
    <div id="cb-avatar">🎬</div>
    <div id="cb-info">
      <div id="cb-name">CineBot</div>
      <div id="cb-status"><span id="cb-dot"></span>Online &nbsp;·&nbsp; Movie Expert</div>
    </div>
    <button id="cb-close">✕</button>
  </div>
  <div id="cb-msgs">
    <div class="cb-msg b">
      <div class="cb-bbl">Hey! 🎬 I'm CineBot. Ask me to recommend movies by mood, genre, director, year — anything!</div>
      <div class="cb-ts">Just now</div>
    </div>
  </div>
  <div id="cb-typing"><span></span><span></span><span></span></div>
  <div id="cb-nokey">
    <div style="font-size:32px;margin-bottom:8px;">⚠️</div>
    <div style="color:#ffb347;font-weight:600;margin-bottom:6px;">OpenRouter key not set</div>
    <div>Add <code>OPENROUTER_API_KEY</code> to your <code>.env</code> file.</div>
    <div style="margin-top:10px;">Get a free key at:<br>
      <a href="https://openrouter.ai/keys" target="_blank">openrouter.ai/keys</a></div>
  </div>
  <div id="cb-qp">
    <div class="cb-q">🌑 Dark thriller</div>
    <div class="cb-q">😄 Feel-good pick</div>
    <div class="cb-q">⭐ Best rated</div>
    <div class="cb-q">🎬 Nolan films</div>
    <div class="cb-q">🚀 Sci-fi classic</div>
    <div class="cb-q">😢 Emotional drama</div>
  </div>
  <div id="cb-inp-area">
    <textarea id="cb-inp" placeholder="Ask about movies…" rows="1"></textarea>
    <button id="cb-snd">&#10148;</button>
  </div>
</div>
"""

    # Build the full injection script
    script = f"""
(function() {{
  'use strict';
  
  var pd = window.parent.document;
  var pw = window.parent;

  // Inject only once per session
  if (pd.getElementById('cinebot-fab')) return;

  // ── Inject CSS into parent <head> ──
  var style = pd.createElement('style');
  style.id = 'cinebot-style';
  style.textContent = {json.dumps(CSS)};
  pd.head.appendChild(style);

  // ── Inject HTML into parent <body> ──
  var host = pd.createElement('div');
  host.id = 'cinebot-host';
  host.innerHTML = {json.dumps(PANEL_HTML)};
  pd.body.appendChild(host);

  // ── Wait for DOM elements to be ready ──
  function getElement(id) {{
    var el = pd.getElementById(id);
    if (!el) console.error('Element not found: ' + id);
    return el;
  }}

  // ── Constants ──
  var KEY     = {json.dumps(api_key)};
  var HAS_KEY = {json.dumps(has_key)};
  var MOVIES  = {movies_json};
  var cbOpen  = false;
  var cbHist  = [];

  console.log('CineBot initialized. HAS_KEY:', HAS_KEY);

  // ── No-key state ──
  if (!HAS_KEY) {{
    var msgs = getElement('cb-msgs');
    var typing = getElement('cb-typing');
    var qp = getElement('cb-qp');
    var inp_area = getElement('cb-inp-area');
    if (msgs) msgs.style.display = 'none';
    if (typing) typing.style.display = 'none';
    if (qp) qp.style.display = 'none';
    if (inp_area) inp_area.style.display = 'none';
    var nokey = getElement('cb-nokey');
    if (nokey) nokey.style.display = 'flex';
  }}

  // ── System prompt ──
  var movieLines = MOVIES.slice(0, 250).map(function(m) {{
    return m.t + ' (' + m.y + ') | ' + m.g + ' | Rating:' + m.r + ' | Dir:' + m.d + ' | Mood:' + m.m;
  }}).join('\\n');
  
  var SYS = 'You are CineBot, a friendly expert movie assistant for CineMatch. '
    + 'Recommend movies from the dataset provided. Always include title, year and rating. '
    + 'Give 4-6 picks as a numbered list, each with a one-line reason. '
    + 'Keep replies under 180 words. Be enthusiastic like a movie-loving friend.\\n\\n'
    + 'MOVIE DATASET:\\n' + movieLines;

  // ── Toggle panel ──
  function cbToggle() {{
    cbOpen = !cbOpen;
    var panel = getElement('cinebot-panel');
    var fab = getElement('cinebot-fab');
    if (panel) {{
      if (cbOpen) {{
        panel.classList.add('cb-open');
      }} else {{
        panel.classList.remove('cb-open');
      }}
    }}
    if (fab) fab.innerHTML = cbOpen ? '✕' : '🎬';
    if (cbOpen) cbScroll();
  }}

  function cbScroll() {{
    var m = getElement('cb-msgs');
    if (m) setTimeout(function() {{ m.scrollTop = m.scrollHeight; }}, 60);
  }}

  function cbNow() {{
    return new Date().toLocaleTimeString([], {{hour:'2-digit', minute:'2-digit'}});
  }}

  function cbEsc(s) {{
    return s
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/\\*\\*(.*?)\\*\\*/g,'<b>$1</b>')
      .replace(/^(\\d+\\.)/gm,'<span style="color:#e50914;font-weight:700">$1</span>')
      .replace(/\\n/g,'<br>');
  }}

  function cbAddMsg(role, text) {{
    var msgs = getElement('cb-msgs');
    if (!msgs) return;
    var wrap = pd.createElement('div');
    wrap.className = 'cb-msg ' + role;
    wrap.innerHTML = '<div class="cb-bbl">' + cbEsc(text) + '</div>'
                   + '<div class="cb-ts">' + cbNow() + '</div>';
    msgs.appendChild(wrap);
    cbScroll();
  }}

  function cbTyping(show) {{
    var t = getElement('cb-typing');
    var m = getElement('cb-msgs');
    if (show) {{
      if (t) t.style.display = 'flex';
      if (m) m.appendChild(t);
      cbScroll();
    }} else {{
      if (t) t.style.display = 'none';
    }}
  }}

  function cbHideQP() {{
    var qp = getElement('cb-qp');
    if (qp) qp.style.display = 'none';
  }}

  async function cbCallAPI(messages) {{
    var models = [
      'qwen/qwen3.6-plus:free',
      'meta-llama/llama-3.3-70b-instruct:free',
      'google/gemma-3-12b-it:free'
    ];

    for (var i = 0; i < models.length; i++) {{
      try {{
        var resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {{
          method: 'POST',
          headers: {{
            'Authorization': 'Bearer ' + KEY,
            'Content-Type': 'application/json',
            'HTTP-Referer': pw.location.href,
            'X-Title': 'CineMatch CineBot'
          }},
          body: JSON.stringify({{
            model: models[i],
            max_tokens: 480,
            temperature: 0.75,
            messages: [{{role:'system', content:SYS}}].concat(messages)
          }})
        }});
        var data = await resp.json();
        if (data.choices && data.choices[0]) {{
          var content = data.choices[0].message.content || '';
          content = content.replace(/<think>[\\s\\S]*?<\\/think>/gi, '').trim();
          return content;
        }}
        
        // If it failed and there are no more fallbacks left
        if (i === models.length - 1) {{
          if (data.error) return '❌ ' + (data.error.message || 'API error');
          return '❌ Unexpected response.';
        }}
        
        console.warn('CineBot: ' + models[i] + ' failed (' + (data.error ? data.error.message : 'Unknown') + '). Falling back...');
      }} catch(e) {{
        if (i === models.length - 1) return '❌ Network error: ' + e.message;
        console.warn('CineBot: Network fetch failed for ' + models[i] + '. Falling back...');
      }}
    }}
  }}

  async function cbSend() {{
    if (!HAS_KEY) {{
      alert('Please add OPENROUTER_API_KEY to your .env file');
      return;
    }}
    var inp  = getElement('cb-inp');
    var snd  = getElement('cb-snd');
    if (!inp || !snd) return;
    var text = inp.value.trim();
    if (!text) return;
    inp.value = '';
    inp.style.height = 'auto';
    
    // Lock input immediately
    inp.disabled = true;
    snd.disabled = true;
    
    cbHideQP();
    cbAddMsg('u', text);
    cbHist.push({{role:'user', content:text}});
    
    // Show loading message with dots animation
    cbAddLoadingMsg();
    
    var reply = await cbCallAPI(cbHist);
    
    // Remove loading message
    cbRemoveLoadingMsg();
    
    var role = reply.startsWith('❌') ? 'err' : 'b';
    cbAddMsg(role, reply);
    if (role === 'b') cbHist.push({{role:'assistant', content:reply}});
    
    // Unlock input
    inp.disabled = false;
    snd.disabled = false;
    inp.focus();
  }}

  function cbAddLoadingMsg() {{
    var msgs = getElement('cb-msgs');
    if (!msgs) return;
    var wrap = pd.createElement('div');
    wrap.id = 'cb-loading-msg';
    wrap.className = 'cb-msg b';
    wrap.innerHTML = '<div class="cb-bbl cb-loading"><span></span><span></span><span></span></div>';
    msgs.appendChild(wrap);
    cbScroll();
  }}

  function cbRemoveLoadingMsg() {{
    var loading = pd.getElementById('cb-loading-msg');
    if (loading) loading.remove();
  }}

  function cbSQ(el) {{
    var inp = getElement('cb-inp');
    if (!inp) return;
    var text = el.textContent.trim().replace(/^\\S+\\s/, '');
    inp.value = text;
    cbSend();
  }}

  function cbKey(e) {{
    if (e.key === 'Enter' && !e.shiftKey) {{ 
      e.preventDefault(); 
      cbSend(); 
    }}
  }}

  // ── Attach event listeners ──
  var fab = getElement('cinebot-fab');
  var closeBtn = getElement('cb-close');
  var sendBtn = getElement('cb-snd');
  var inp = getElement('cb-inp');

  if (fab) fab.addEventListener('click', cbToggle);
  if (closeBtn) closeBtn.addEventListener('click', cbToggle);
  if (sendBtn) sendBtn.addEventListener('click', cbSend);
  if (inp) {{
    inp.addEventListener('keydown', cbKey);
    inp.addEventListener('input', function() {{
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 72) + 'px';
    }});
  }}

  // ── Attach question prompt listeners ──
  var questions = pd.querySelectorAll('.cb-q');
  questions.forEach(function(q) {{
    q.addEventListener('click', function() {{ cbSQ(this); }});
  }});

  console.log('CineBot ready. FAB click listener attached.');
}})();
"""

    # Wrap in a minimal HTML page that just runs the injection script
    iframe_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:transparent;overflow:hidden;">
<script>{script}</script>
</body></html>"""

    components.html(iframe_html, height=0, scrolling=False)