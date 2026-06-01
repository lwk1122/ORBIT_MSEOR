import asyncio
import json
import os
from datetime import datetime, timezone
from functools import partial

from aiohttp import web

from . import analytics
from . import api
from .logging_utils import configure_logging


def _parse_plan_items(plan):
    if not plan:
        return []
    try:
        items = json.loads(plan.get("content") or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _phase_for_round(round_number):
    if not round_number or round_number <= 1:
        return "opening"
    if round_number == 2:
        return "evidence"
    return "analysis"


def build_dashboard_snapshot(topic_id=None, subtopic_id=None):
    topic = api.get_topic(int(topic_id)) if topic_id is not None else api.get_current_topic()
    if not topic:
        return {
            "topic": None,
            "plan": None,
            "subtopics": [],
            "current_subtopic": None,
            "messages": [],
            "facts": [],
            "claims": [],
            "fact_candidates": [],
            "claim_candidates": [],
            "votes": [],
            "web_evidence": [],
            "ledger_entries": [],
            "contested_pairs": [],
            "knowledge_edges": [],
            "ledger_edges": [],
            "ledger_pending": [],
            "code_evidence": [],
            "mse_review": None,
            "status": {
                "db_path": os.path.basename(api.get_db_path()),
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "current_round": None,
                "current_phase": None,
            },
        }

    topic_id = topic["id"]
    plan = api.get_active_plan(topic_id)
    subtopics = api.get_current_subtopics(topic_id)
    if subtopic_id:
        current_subtopic = api.get_subtopic(int(subtopic_id))
        if current_subtopic is None:
            current_subtopic = api.get_open_subtopic(
                topic_id
            ) or api.get_latest_subtopic(topic_id)
    else:
        current_subtopic = api.get_open_subtopic(topic_id) or api.get_latest_subtopic(
            topic_id
        )
    messages = api.get_messages(
        topic_id,
        subtopic_id=current_subtopic["id"] if current_subtopic else None,
        limit=2000,
    )
    facts = api.get_facts(topic_id, limit=2000)
    claims = api.get_claims(topic_id, limit=2000)
    fact_candidates = []
    claim_candidates = []
    if current_subtopic:
        fact_candidates = api.get_fact_candidates(
            topic_id, subtopic_id=current_subtopic["id"], limit=10000
        )
        claim_candidates = api.get_claim_candidates(
            topic_id, subtopic_id=current_subtopic["id"], limit=10000
        )
    votes = api.get_vote_records(
        topic_id,
        subtopic_id=current_subtopic["id"] if current_subtopic else None,
        limit=1000,
    )

    last_round = None
    for message in reversed(messages):
        if message.get("round_number") is not None:
            last_round = message["round_number"]
            break

    return {
        "topic": topic,
        "plan": (
            {
                "id": plan["id"],
                "current_index": plan["current_index"],
                "items": _parse_plan_items(plan),
            }
            if plan
            else None
        ),
        "subtopics": subtopics,
        "current_subtopic": current_subtopic,
        "messages": messages,
        "facts": facts,
        "claims": claims,
        "fact_candidates": fact_candidates,
        "claim_candidates": claim_candidates,
        "votes": votes,
        "web_evidence": api.get_web_evidence_for_topic(topic_id),
        "code_evidence": api.get_code_evidence_for_topic_full(topic_id),
        "ledger_entries": api.get_ledger_entries_with_names(topic_id),
        "contested_pairs": api.get_contested_ledger_pairs(topic_id),
        "ledger_edges": api.get_ledger_edges(topic_id),
        "knowledge_edges": api.get_knowledge_edges(topic_id),
        "ledger_pending": api.get_active_ledger_pending(topic_id, last_round or 0),
        "mse_review": api.get_mse_review_snapshot(topic_id),
        "status": {
            "db_path": os.path.basename(api.get_db_path()),
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "current_round": last_round,
            "current_phase": (
                _phase_for_round(last_round) if last_round is not None else None
            ),
        },
    }


def render_dashboard_html():
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ORBIT Monitor</title>
  <link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=JetBrains+Mono:wght@400;700&family=Work+Sans:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #141517;
      --panel: #232529;
      --ink: #e0e0e0;
      --muted: #8b8f98;
      --line: #3a3f4b;
      --accent: #00e5bf;
      --warning: #ffb400;
      --fact: #4da6ff;
      --pending: #e59866;
      --mono: "JetBrains Mono", monospace;
      --sans: "Work Sans", sans-serif;
      --display: "Archivo Black", sans-serif;
      --chat-sys-bg: #1b2633;
      --chat-sys-border: #2c4463;
      --chat-usr-bg: #2a2d32;
      --chat-usr-border: #3a3f4b;
      --chat-dog-border: #d33c46;
      --chat-cat-border: #3cb878;
    }
    
    * { box-sizing: border-box; }
    
    body {
      margin: 0;
      background-color: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden; /* Prevent body scroll */
    }
    
    .header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }
    
    .header-left h1 {
      margin: 0 0 4px 0;
      font-family: var(--display);
      font-size: 20px;
      letter-spacing: -0.02em;
      color: #fff;
      display: -webkit-box;
      -webkit-line-clamp: 2; /* Limit to 2 lines */
      -webkit-box-orient: vertical;
      overflow: hidden;
      max-width: 900px;
      line-height: 1.3;
    }
    
    .eyebrow {
      font-family: var(--mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--accent);
    }
    
    .meta {
      display: flex;
      gap: 16px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
    }
    
    .meta span {
      background: #1a1c20;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 4px;
    }

    .nav-bar select {
      padding: 6px;
      font-family: var(--mono);
      font-size: 12px;
      background: var(--bg);
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 4px;
      outline: none;
    }
    
    .main-container {
      display: grid;
      grid-template-columns: 320px 1fr 380px;
      gap: 1px;
      flex: 1;
      background: var(--line); /* acts as borders between columns */
      overflow: hidden;
    }
    
    .column {
      background: var(--bg);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      height: 100%;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted); }
    
    .panel-header {
      position: sticky;
      top: 0;
      background: var(--panel);
      padding: 12px 16px;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      border-bottom: 1px solid var(--line);
      z-index: 10;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .panel-body {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    
    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
      font-size: 11px;
      font-family: var(--mono);
      color: var(--muted);
    }
    .controls select, .controls input {
      background: var(--bg);
      border: 1px solid var(--line);
      color: var(--ink);
      font-family: var(--mono);
      border-radius: 4px;
      padding: 2px 4px;
      outline: none;
    }
    /* Chat Timeline Styles */
    .timeline {
      padding: 24px;
      gap: 20px;
      align-items: flex-start;
    }
    
    .chat-bubble {
      padding: 14px;
      border-radius: 6px;
      width: 100%;
      max-width: 90%;
      background: var(--chat-usr-bg);
      border: 1px solid var(--chat-usr-border);
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .chat-bubble.system {
      background: var(--chat-sys-bg);
      border-color: var(--chat-sys-border);
    }
    
    .chat-bubble.system .chat-sender { color: var(--accent); }
    .chat-bubble.special-dog { border-left: 4px solid var(--chat-dog-border); }
    .chat-bubble.special-cat { border-left: 4px solid var(--chat-cat-border); }
    
    .chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-family: var(--mono);
      font-size: 11px;
      border-bottom: 1px dashed var(--line);
      padding-bottom: 6px;
    }
    
    .chat-sender { 
      font-weight: 700; 
      text-transform: uppercase; 
      color: #fff;
    }
    .chat-meta { color: var(--muted); }
    
    .chat-content {
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 14px;
    }
    
    details {
      background: rgba(0,0,0,0.2);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 8px;
    }
    
    details > summary {
      cursor: pointer;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--accent);
      font-weight: bold;
      outline: none;
    }
    
    details[open] summary {
      margin-bottom: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }
    
    /* Cards for side panels */
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 12px;
    }
    
    .label {
      display: inline-block;
      padding: 2px 6px;
      background: #1a1c20;
      border: 1px solid var(--line);
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
      margin-right: 6px;
      margin-bottom: 6px;
      border-radius: 2px;
    }
    
    .card-title { font-weight: 600; margin-bottom: 6px; font-size: 13px; color: #fff;}
    .card-content { white-space: pre-wrap; line-height: 1.5; font-size: 13px; color: #ccc;}
    .fact-content { color: var(--fact); font-family: var(--mono); font-size: 12px;}
    .candidate-content { color: var(--pending); font-size: 13px;}
    .empty { color: var(--muted); font-style: italic; font-size: 13px; text-align: center; padding: 20px;}
    
    /* Tabs for KB */
    .tabs {
      display: flex;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 45px;
      background: var(--bg);
      z-index: 9;
    }
    .tab {
      flex: 1;
      text-align: center;
      padding: 10px;
      font-family: var(--mono);
      font-size: 11px;
      cursor: pointer;
      color: var(--muted);
      border-bottom: 2px solid transparent;
    }
    .tab:hover { color: #fff; }
    .tab.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }
    .tab-content { display: none; }
    .tab-content.active { display: flex; flex-direction: column; gap: 12px; padding: 16px; }

    /* Tooltip styles */
    .citation {
      color: var(--accent);
      text-decoration: underline dotted;
      cursor: help;
      font-weight: bold;
      position: relative;
    }
    #global-tooltip {
      position: fixed;
      display: none;
      background: var(--panel);
      border: 1px solid var(--accent);
      padding: 12px;
      border-radius: 4px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      z-index: 10000;
      max-width: 400px;
      font-size: 13px;
      line-height: 1.5;
      pointer-events: none;
      color: #fff;
    }
    #global-tooltip .tt-label {
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 4px;
      display: block;
      border-bottom: 1px solid var(--line);
      padding-bottom: 4px;
    }

    .source-kind { font-size:10px; padding:1px 6px; border-radius:3px; font-family:var(--mono); }
    .sk-web { background:#1a3a2a; color:#3cb878; }
    .sk-code { background:#2a1a3a; color:#a78bfa; }
    .sk-agent { background:#2a2a2a; color:var(--muted); }

    .edge-list { font-size:11px; color:var(--muted); margin-top:4px; }
    .edge-list .edge-in { color:#60a5fa; }
    .edge-list .edge-out { color:#f59e0b; }
    .edge-rel { font-size:10px; padding:0 4px; border-radius:2px; background:var(--line); }

    .fact-superseded { text-decoration: line-through; opacity: 0.6; }
  </style>
</head>
<body>
  <div id="global-tooltip"></div>
  <div class="header">
    <div class="header-left">
      <div class="eyebrow">ORBIT Monitor</div>
      <h1 id="topic-title">Loading...</h1>
    </div>
    <div class="header-right" style="display:flex; align-items:center; gap:20px;">
      <div class="nav-bar" id="topic-nav" style="display:block;">
        <select id="topic-select" onchange="changeTopic(this.value)"></select>
      </div>
      <div class="nav-bar" id="nav-bar" style="display:none;">
        <select id="subtopic-select" onchange="changeSubtopic(this.value)"></select>
      </div>
      <a id="graph-link" href="/graph" target="_blank" rel="noopener" style="color:var(--accent);font-size:12px;text-decoration:none;border:1px solid var(--accent);padding:4px 10px;border-radius:4px;white-space:nowrap;">Knowledge Graph</a>
      <a id="report-link" href="#" target="_blank" rel="noopener" style="display:none;color:var(--accent);font-size:12px;text-decoration:none;border:1px solid var(--accent);padding:4px 10px;border-radius:4px;white-space:nowrap;margin-left:6px;">Report</a>
      <div class="meta" id="topic-meta"></div>
    </div>
  </div>
  
  <div class="main-container">
    <!-- Left Column: Plan & Details -->
    <div class="column">
      <div class="panel-header">Topic Details</div>
      <div class="panel-body">
        <div id="topic-detail" class="card-content" style="margin-bottom:20px;"></div>
        <div id="hitl-panel" style="display:none;"></div>
        <div id="plan-panel" style="display:flex;flex-direction:column;gap:12px;"></div>
      </div>
    </div>
    
    <!-- Middle Column: Timeline -->
    <div class="column" id="scroll-timeline">
      <div class="panel-header" style="flex-wrap: wrap; gap: 8px;">
        <div>
            <span>Timeline</span>
            <span id="timeline-stats" style="color:var(--muted);font-weight:normal; margin-left: 8px;"></span>
        </div>
        <div class="controls">
            <label><input type="checkbox" id="auto-refresh-toggle" checked onchange="toggleAutoRefresh(this)"> Auto-Refresh</label>
            <select id="sort-toggle" onchange="toggleSort(this.value)">
                <option value="asc">Oldest First</option>
                <option value="desc">Newest First</option>
            </select>
        </div>
      </div>
      <div class="panel-body timeline" id="messages-panel"></div>
    </div>
    
    <!-- Right Column: Knowledge Base -->
    <div class="column">
      <div class="panel-header">Knowledge Base</div>
      <div class="tabs">
        <div class="tab active" onclick="switchTab('facts', this)">Facts</div>
        <div class="tab" onclick="switchTab('claims', this)">Claims</div>
        <div class="tab" onclick="switchTab('code', this)">Code</div>
        <div class="tab" onclick="switchTab('mse', this)">MSE</div>
        <div class="tab" onclick="switchTab('ledger', this)">Ledger</div>
        <div class="tab" onclick="switchTab('cands', this)">Pending</div>
        <div class="tab" onclick="switchTab('web', this)">Web</div>
      </div>

      <div id="tab-facts" class="tab-content active"></div>
      <div id="tab-claims" class="tab-content"></div>
      <div id="tab-code" class="tab-content"></div>
      <div id="tab-mse" class="tab-content"></div>
      <div id="tab-ledger" class="tab-content"></div>
      <div id="tab-cands" class="tab-content"></div>
      <div id="tab-web" class="tab-content"></div>
    </div>
  </div>

  <script>
    const KNOWLEDGE_MAP = {};

    function updateKnowledgeMap(snapshot) {
      for (const key in KNOWLEDGE_MAP) delete KNOWLEDGE_MAP[key];
      if (snapshot.facts) snapshot.facts.forEach(f => KNOWLEDGE_MAP['F' + f.id] = { type: 'Fact', content: f.content });
      if (snapshot.claims) snapshot.claims.forEach(c => KNOWLEDGE_MAP['C' + c.id] = { type: 'Claim', content: c.content });
      if (snapshot.web_evidence) snapshot.web_evidence.forEach(w => KNOWLEDGE_MAP['W' + w.id] = { type: 'Web Source', content: w.title + ': ' + w.snippet });
      if (snapshot.ledger_entries) snapshot.ledger_entries.forEach(l => {
        let desc = l.entity_name + ' | ' + l.attribute_name + ': ' + l.value;
        if (l.unit) desc += ' ' + l.unit;
        if (l.normalized_timeframe) desc += ' (' + l.normalized_timeframe + ')';
        if (l.source_ref) desc += ' [' + l.source_ref + ']';
        KNOWLEDGE_MAP['L' + l.id] = { type: 'Ledger', content: desc };
      });
      if (snapshot.code_evidence) snapshot.code_evidence.forEach(e => KNOWLEDGE_MAP['E' + e.id] = { type: 'Code Evidence', content: (e.success ? 'PASSED' : 'FAILED') + ': ' + e.hypothesis });
      if (snapshot.mse_review && snapshot.mse_review.documents) {
        snapshot.mse_review.documents.forEach(doc => {
          (doc.chunks || []).forEach(ch => {
            KNOWLEDGE_MAP['D' + ch.id] = {
              type: 'Corpus Chunk',
              content: (doc.title || doc.source_path || 'Document') + ': ' + (ch.text || ch.table_markdown || '').substring(0, 300)
            };
          });
        });
      }
      if (snapshot.messages) snapshot.messages.forEach(m => KNOWLEDGE_MAP['M' + m.id] = { type: 'Message', content: m.sender + ': ' + (m.content || '').substring(0, 200) + ((m.content || '').length > 200 ? '...' : '') });
    }

    // --- Edge Index ---
    const EDGE_INDEX = {};
    function buildEdgeIndex(snapshot) {
      for (const key in EDGE_INDEX) delete EDGE_INDEX[key];
      (snapshot.knowledge_edges || []).forEach(e => {
        const srcKey = _typePrefix(e.source_type) + e.source_id;
        const tgtKey = _typePrefix(e.target_type) + e.target_id;
        if (!EDGE_INDEX[srcKey]) EDGE_INDEX[srcKey] = [];
        if (!EDGE_INDEX[tgtKey]) EDGE_INDEX[tgtKey] = [];
        EDGE_INDEX[srcKey].push({relation: e.relation, peer: tgtKey, dir: 'out', confidence: e.confidence});
        EDGE_INDEX[tgtKey].push({relation: e.relation, peer: srcKey, dir: 'in', confidence: e.confidence});
      });
    }
    function _typePrefix(t) {
      return {fact:'F', claim:'C', web_evidence:'W', code_evidence:'E', ledger:'L', tool_trace:'T'}[t] || '?';
    }

    function _sourceKindSpan(sk) {
      if (!sk) return '';
      const cls = sk === 'web' ? 'sk-web' : (sk === 'code' ? 'sk-code' : 'sk-agent');
      return '<span class="source-kind ' + cls + '">' + esc(sk) + '</span>';
    }

    function _renderEdgeDetails(key, detailsId) {
      const edges = EDGE_INDEX[key];
      if (!edges || !edges.length) return '';
      // Group by relation
      const groups = {};
      edges.forEach(e => {
        const gk = (e.dir === 'in' ? '\u2190 ' : '\u2192 ') + e.relation;
        if (!groups[gk]) groups[gk] = [];
        groups[gk].push(e);
      });
      let html = '<details id="' + esc(detailsId) + '" class="edge-list" style="margin-top:6px;"><summary style="font-size:11px;cursor:pointer;">' + edges.length + ' edge(s)</summary><div style="margin-top:4px;">';
      Object.keys(groups).forEach(gk => {
        html += '<div style="margin-bottom:2px;"><span class="edge-rel">' + esc(gk) + '</span> ';
        html += groups[gk].map(e => {
          const cls = e.dir === 'in' ? 'edge-in' : 'edge-out';
          return '<span class="citation ' + cls + '" onmouseover="showTooltip(event, \'' + esc(e.peer) + '\')" onmouseout="hideTooltip()" style="cursor:help;">[' + esc(e.peer) + ']</span>';
        }).join(' ');
        html += '</div>';
      });
      html += '</div></details>';
      return html;
    }

    function linkCitations(text) {
      return esc(text).replace(/\[([DFWCLMETA])(\d+)\]/g, (match, type, id) => {
        const key = type + id;
        return '<span class="citation" onmouseover="showTooltip(event, \'' + key + '\')" onmouseout="hideTooltip()">[' + key + ']</span>';
      });
    }

    function showTooltip(e, key) {
      const tt = document.getElementById('global-tooltip');
      const data = KNOWLEDGE_MAP[key];
      if (!data) return;

      let content = '<span class="tt-label">' + esc(data.type) + ' [' + key + ']</span>' + esc(data.content);

      // Append edge relations if present
      const edges = EDGE_INDEX[key];
      if (edges && edges.length) {
        content += '<div style="margin-top:8px;border-top:1px solid var(--line);padding-top:6px;font-size:11px;color:var(--muted);">';
        content += '<div style="font-family:var(--mono);margin-bottom:4px;">Relations</div>';
        edges.forEach(edge => {
          const arrow = edge.dir === 'in' ? '\u2190' : '\u2192';
          const cls = edge.dir === 'in' ? 'edge-in' : 'edge-out';
          content += '<div><span class="' + cls + '">' + arrow + '</span> <span class="edge-rel">' + esc(edge.relation) + '</span> [' + esc(edge.peer) + ']</div>';
        });
        content += '</div>';
      }

      tt.innerHTML = content;
      tt.style.display = 'block';

      // Positioning
      const x = e.clientX + 15;
      const y = e.clientY + 15;
      tt.style.left = x + 'px';
      tt.style.top = y + 'px';

      // Flip if overflow
      const rect = tt.getBoundingClientRect();
      if (rect.right > window.innerWidth) tt.style.left = (e.clientX - rect.width - 15) + 'px';
      if (rect.bottom > window.innerHeight) tt.style.top = (e.clientY - rect.height - 15) + 'px';
    }

    function hideTooltip() {
      document.getElementById('global-tooltip').style.display = 'none';
    }

    
    let isAutoRefresh = true;
    let sortOrder = 'asc';

    function toggleAutoRefresh(el) {
        isAutoRefresh = el.checked;
        if(isAutoRefresh) refresh();
    }
    
    function toggleSort(val) {
        sortOrder = val;
        refresh(); // force re-render immediately
    }

    let currentTopicId = new URLSearchParams(window.location.search).get("topic_id");
    let currentSubtopicId = new URLSearchParams(window.location.search).get("subtopic_id");

    async function loadTopicOptions(selectedTopicId) {
        if (window.ORBIT_STATIC_DATA) return;
        try {
            const response = await fetch('/api/topics', { cache: 'no-store' });
            if (!response.ok) return;
            const data = await response.json();
            const topicSelect = document.getElementById('topic-select');
            const topics = data.topics || [];
            const activeId = String(selectedTopicId || currentTopicId || data.current_topic_id || '');
            if (!topics.length) {
                topicSelect.innerHTML = '<option value="">-- No Topics --</option>';
                return;
            }
            topicSelect.innerHTML = topics.map(t => {
                const selected = String(t.id) === activeId ? 'selected' : '';
                return '<option value="' + esc(t.id) + '" ' + selected + '>#' + esc(t.id) + ' ' + esc((t.summary || '').substring(0, 60)) + '</option>';
            }).join('');
            currentTopicId = activeId || String(topics[0].id);
        } catch (err) {
            console.warn('Failed to load topics', err);
        }
    }

    function _updateLocation(topicId, subtopicId) {
        const params = new URLSearchParams(window.location.search);
        if (topicId) params.set('topic_id', topicId);
        else params.delete('topic_id');
        if (subtopicId) params.set('subtopic_id', subtopicId);
        else params.delete('subtopic_id');
        const qs = params.toString();
        window.location.href = qs ? ('/?' + qs) : '/';
    }

    function changeTopic(id) {
        currentTopicId = id || null;
        currentSubtopicId = null;
        _updateLocation(currentTopicId, null);
    }

    function changeSubtopic(id) {
        if (window.ORBIT_STATIC_DATA) {
            currentSubtopicId = id;
            refresh();
        } else {
            _updateLocation(currentTopicId, id || null);
        }
    }

    function resumeTopic(topicId) {
        if (!confirm('Resume this topic?')) return;
        fetch('/api/topic/' + topicId + '/resume', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.ok) { refresh(); }
                else { alert('Resume failed: ' + (data.error || 'unknown')); }
            })
            .catch(err => alert('Resume error: ' + err));
    }

    function injectKnowledge(topicId) {
        const typeEl = document.getElementById('inject-type');
        const contentEl = document.getElementById('inject-content');
        if (!typeEl || !contentEl || !contentEl.value.trim()) {
            alert('Please enter content to inject.');
            return;
        }
        fetch('/api/topic/' + topicId + '/inject', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: typeEl.value, content: contentEl.value.trim() })
        })
            .then(r => r.json())
            .then(data => {
                if (data.injection_id) {
                    contentEl.value = '';
                    refresh();
                } else {
                    alert('Inject failed: ' + (data.error || 'unknown'));
                }
            })
            .catch(err => alert('Inject error: ' + err));
    }

    function reviewComponent(componentId, status) {
        fetch('/api/mse/component/' + componentId + '/review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ review_status: status })
        })
            .then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(result => {
                if (result.ok && result.data.ok) refresh();
                else alert('Component review failed: ' + (result.data.error || 'unknown'));
            })
            .catch(err => alert('Component review error: ' + err));
    }

    function resolveDiagnostic(diagnosticId, status) {
        fetch('/api/mse/diagnostic/' + diagnosticId + '/status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ status: status || 'resolved', resolution: 'Marked from monitor' })
        })
            .then(r => r.json().then(data => ({ ok: r.ok, data })))
            .then(result => {
                if (result.ok && result.data.ok) refresh();
                else alert('Diagnostic update failed: ' + (result.data.error || 'unknown'));
            })
            .catch(err => alert('Diagnostic update error: ' + err));
    }

    function switchTab(tabName, tabEl) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tabEl.classList.add('active');
      document.getElementById('tab-' + tabName).classList.add('active');
    }

    function jumpToFact(fid) {
      const tabBtn = document.querySelector('.tab[onclick*="facts"]');
      if (tabBtn) { switchTab('facts', tabBtn); }
      setTimeout(() => {
        const el = document.getElementById('fact-' + fid);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          el.style.outline = '2px solid var(--fact)';
          setTimeout(() => { el.style.outline = ''; }, 2000);
        }
      }, 50);
    }

    function esc(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function renderEmpty(node, text) {
      node.innerHTML = '<div class="empty">' + esc(text) + '</div>';
    }

    function renderTopic(snapshot) {
      const title = document.getElementById('topic-title');
      const detail = document.getElementById('topic-detail');
      const meta = document.getElementById('topic-meta');
      const navBar = document.getElementById('nav-bar');
      const subSelect = document.getElementById('subtopic-select');
      const graphLink = document.getElementById('graph-link');
      const topicSelect = document.getElementById('topic-select');
      
      if (!snapshot.topic) {
        title.textContent = 'No active topic';
        meta.innerHTML = '<span>No data</span>';
        return;
      }

      currentTopicId = String(snapshot.topic.id);
      if (topicSelect) topicSelect.value = currentTopicId;
      if (graphLink) graphLink.href = '/graph?topic_id=' + encodeURIComponent(snapshot.topic.id);
      
      // Let CSS handle text wrapping instead of hard truncation
      let displayTitle = snapshot.topic.summary;
      title.textContent = displayTitle;
      title.title = snapshot.topic.summary; // full text on hover
      
      
      
      const topicDetail = snapshot.topic.detail || '';
      let detailHtml = topicDetail
          ? '<details id="topic-detail-expand"><summary style="font-size:13px;cursor:pointer;">' + esc(snapshot.topic.summary) + '</summary><div style="margin-top:8px;">' + esc(topicDetail) + '</div></details>'
          : esc(snapshot.topic.summary);
      if (snapshot.topic.conclusion) {
          detailHtml += '<div style="margin-top: 16px; padding-top: 16px; border-top: 1px dashed var(--line);">';
          detailHtml += '<span class="eyebrow" style="display:block; margin-bottom:8px; color:var(--warning);">Final Topic Conclusion:</span>';
          detailHtml += '<strong style="color:#fff;">' + esc(snapshot.topic.conclusion) + '</strong></div>';
      }
      if (snapshot.current_subtopic && snapshot.current_subtopic.conclusion) {
          detailHtml += '<div style="margin-top: 16px; padding-top: 16px; border-top: 1px dashed var(--line);">';
          detailHtml += '<span class="eyebrow" style="display:block; margin-bottom:8px; color:var(--accent);">Subtopic Conclusion:</span>';
          detailHtml += '<strong style="color:#fff;">' + esc(snapshot.current_subtopic.conclusion) + '</strong></div>';
      }
      if (snapshot.current_subtopic && snapshot.current_subtopic.locked_scope) {
          try {
              const scope = JSON.parse(snapshot.current_subtopic.locked_scope);
              let scopeHtml = '<div style="margin-top: 12px; padding: 8px 12px; border-left: 3px solid var(--warning); background: rgba(255,200,0,0.05); border-radius: 4px;">';
              scopeHtml += '<span class="eyebrow" style="display:block; margin-bottom:4px; color:var(--warning);">Locked Scope</span>';
              if (scope.target_metric) scopeHtml += '<div style="font-size:12px; color:var(--text);"><strong>Metric:</strong> ' + esc(scope.target_metric) + '</div>';
              if (scope.entity_boundaries) scopeHtml += '<div style="font-size:12px; color:var(--text);"><strong>Entities:</strong> ' + esc(scope.entity_boundaries) + '</div>';
              if (scope.metric_definition) scopeHtml += '<div style="font-size:12px; color:var(--text);"><strong>Definition:</strong> ' + esc(scope.metric_definition) + '</div>';
              scopeHtml += '</div>';
              detailHtml += scopeHtml;
          } catch(e) {}
      }
      detail.innerHTML = detailHtml;

      
      const bits = [
        'STATUS:' + snapshot.topic.status,
        'ROUND:' + (snapshot.status.current_round ?? '-'),
        'PHASE:' + (snapshot.status.current_phase ?? '-')
      ];
      meta.innerHTML = bits.map(bit => '<span>' + esc(bit) + '</span>').join('');
      
      if(snapshot.subtopics && snapshot.subtopics.length > 0) {
          navBar.style.display = "block";
          let opts = "<option value=''>-- Latest Subtopic --</option>";
          snapshot.subtopics.forEach(st => {
              let selected = (currentSubtopicId && st.id == currentSubtopicId) || (!currentSubtopicId && snapshot.current_subtopic && st.id == snapshot.current_subtopic.id) ? "selected" : "";
              opts += "<option value='" + esc(st.id) + "' " + selected + ">#" + esc(st.id) + " " + esc(st.summary.substring(0, 30)) + "...</option>";
          });
          subSelect.innerHTML = opts;
      }

      // HITL panel — skip rebuild if user is interacting with the inject form
      const hitlPanel = document.getElementById('hitl-panel');
      const activeEl = document.activeElement;
      const hitlHasFocus = activeEl && (activeEl.id === 'inject-content' || activeEl.id === 'inject-type');
      if (snapshot.topic.status === 'Paused' && hitlHasFocus && hitlPanel.style.display === 'block') {
          // Preserve user input — skip rebuild
      } else if (snapshot.topic.status === 'Paused') {
          const stage = snapshot.topic.paused_at_stage || 'unknown';
          const stageLabels = {
              plan_approval: 'Plan Approval',
              subtopic_review: 'Subtopic Review',
              final_review: 'Final Review',
              replan: 'Replan Approval'
          };
          let h = '<div style="background:rgba(245,158,11,0.08);border:1px solid var(--warning);border-radius:8px;padding:16px;margin-bottom:16px;">';
          h += '<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">';
          h += '<span style="background:var(--warning);color:#000;font-weight:700;font-size:11px;padding:3px 8px;border-radius:4px;">PAUSED</span>';
          h += '<span style="color:var(--warning);font-size:13px;font-weight:600;">' + esc(stageLabels[stage] || stage) + '</span>';
          h += '</div>';
          // Resume button
          h += '<button onclick="resumeTopic(' + snapshot.topic.id + ')" style="background:var(--accent);color:#fff;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-weight:600;font-size:13px;margin-bottom:14px;">Resume Topic</button>';
          // Inject knowledge form
          h += '<div style="margin-top:8px;border-top:1px solid var(--warning);padding-top:12px;">';
          h += '<div style="font-size:12px;font-weight:600;color:var(--fg);margin-bottom:8px;">Inject Knowledge</div>';
          h += '<div style="display:flex;gap:6px;margin-bottom:6px;">';
          h += '<select id="inject-type" style="background:var(--bg);color:var(--fg);border:1px solid var(--line);padding:6px 8px;border-radius:4px;font-size:12px;flex-shrink:0;">';
          h += '<option value="text">Text</option><option value="url">URL</option><option value="search_query">Search</option>';
          h += '</select>';
          h += '<input id="inject-content" type="text" placeholder="Enter text, URL, or search query..." style="flex:1;min-width:0;background:var(--bg);color:var(--fg);border:1px solid var(--line);padding:6px 10px;border-radius:4px;font-size:12px;">';
          h += '<button onclick="injectKnowledge(' + snapshot.topic.id + ')" style="background:var(--warning);color:#000;border:none;padding:6px 12px;border-radius:4px;cursor:pointer;font-weight:600;font-size:12px;white-space:nowrap;flex-shrink:0;">Inject</button>';
          h += '</div>';
          h += '</div>';
          // Pending injections list
          h += '<div id="hitl-injections"></div>';
          h += '</div>';
          hitlPanel.innerHTML = h;
          hitlPanel.style.display = 'block';
          // Load pending injections
          fetch('/api/topic/' + snapshot.topic.id + '/pause_status')
            .then(r => r.json())
            .then(data => {
              const inj = data.pending_injections || [];
              const injNode = document.getElementById('hitl-injections');
              if (injNode && inj.length) {
                injNode.innerHTML = '<div style="margin-top:8px;font-size:11px;color:var(--muted);">Pending injections (' + inj.length + '):</div>' +
                  inj.map(i => '<div style="font-size:11px;color:var(--fg);padding:4px 0;border-bottom:1px solid var(--line);"><span style="color:var(--accent);font-weight:600;">[' + esc(i.injection_type) + ']</span> ' + esc((i.content || '').substring(0, 100)) + '</div>').join('');
              }
            }).catch(() => {});
      } else {
          hitlPanel.style.display = 'none';
          hitlPanel.innerHTML = '';
      }
    }

    function renderPlan(snapshot) {
      const node = document.getElementById('plan-panel');
      if (!snapshot.topic) return;
      
      const items = [];
      snapshot.subtopics.forEach((st) => {
        let statusColor = st.status === 'Open' ? 'var(--accent)' : (st.status === 'Closed' ? 'var(--muted)' : 'var(--warning)');
        let scopeBlock = '';
        if (st.locked_scope) {
            try {
                const sc = JSON.parse(st.locked_scope);
                let parts = [];
                if (sc.target_metric) parts.push('<strong>Metric:</strong> ' + esc(sc.target_metric));
                if (sc.entity_boundaries) parts.push('<strong>Entities:</strong> ' + esc(sc.entity_boundaries));
                if (parts.length) scopeBlock = '<div style="font-size:11px; color:var(--warning); margin-top:4px;">' + parts.join(' · ') + '</div>';
            } catch(e) {}
        }
        const bodyText = st.conclusion || st.detail;
        const bodyHtml = bodyText
            ? '<details id="plan-st-' + st.id + '"><summary style="font-size:12px;cursor:pointer;">Details</summary><div class="card-content">' + esc(bodyText) + '</div></details>'
            : '';
        items.push(
          '<div class="card" style="border-left: 3px solid ' + statusColor + '">' +
          '<div><span class="label">ST #' + st.id + '</span><span class="label" style="color:' + statusColor + '">' + esc(st.status) + '</span></div>' +
          '<div class="card-title">' + esc(st.summary) + '</div>' +
          scopeBlock +
          bodyHtml +
          '</div>'
        );
      });
      if (!items.length) {
        renderEmpty(node, 'Plan not generated yet.');
        return;
      }
      node.innerHTML = items.join('');
    }

    function renderMessages(snapshot) {
      const node = document.getElementById('messages-panel');
      const statsNode = document.getElementById('timeline-stats');
      
      if (!snapshot.messages.length && (!snapshot.votes || !snapshot.votes.length)) {
        renderEmpty(node, 'No messages or votes yet.');
        statsNode.textContent = "";
        return;
      }

      statsNode.textContent = snapshot.messages.length + " Msgs";
      const adminRoles = ["skynet", "librarian", "writer", "tron"];
      
      let html = '';
      let currentRound = null;
      const renderedVoteRounds = new Set();

      function renderVotesForRound(r) {
          if(!snapshot.votes) return '';
          const roundVotes = snapshot.votes.filter(v => v.round_number === r && v.vote_kind === 'termination');
          if(roundVotes.length === 0) return '';
          
          let vHtml = '<div class="chat-bubble system" style="border-color:#bda2f7;">';
          vHtml += '<div class="chat-header"><span class="chat-sender" style="color:#bda2f7">GOVERNANCE VOTE</span><span class="chat-meta">Round ' + r + '</span></div>';
          vHtml += '<details id="vote-det-' + r + '"><summary>View ' + roundVotes.length + ' Termination Votes</summary><div class="chat-content" style="font-family: var(--mono); font-size:12px;">';
          
          roundVotes.forEach(v => {
              const symbol = v.decision === 'yes' || v.decision === 'continue' ? '✅' : (v.decision === 'no' || v.decision === 'close' ? '🛑' : '⚠️');
              vHtml += '<strong style="color:#fff">' + esc(v.voter) + '</strong>: ' + symbol + ' ' + esc(v.decision) + '<br>';
              vHtml += '<span style="color:#aaa">Reason: ' + esc(v.reason) + '</span><br><br>';
          });
          
          vHtml += '</div></details></div>';
          return vHtml;
      }

      
      // Messages come from API usually newest first (DESC) because of `ORDER BY id DESC LIMIT`.
      // The API reverses them to chronological order (ASC).
      let msgsToRender = [...snapshot.messages];
      
      // If we want newest first, we reverse them back.
      if (sortOrder === 'desc') {
          msgsToRender.reverse();
      }

      msgsToRender.forEach((message, index) => {
        const trueIndex = sortOrder === 'asc' ? index + 1 : msgsToRender.length - index;
        const timeStr = message.created_at ? message.created_at.substring(11, 16) : '';

        if (currentRound !== null && message.round_number !== null && message.round_number !== currentRound) {
            if (!renderedVoteRounds.has(currentRound)) {
                html += renderVotesForRound(currentRound);
                renderedVoteRounds.add(currentRound);
            }
        }
        if (message.round_number !== null) {
            currentRound = message.round_number;
        }

        let classes = "chat-bubble";
        if(adminRoles.includes(message.sender.toLowerCase())) {
            classes += " system";
        } else if (message.sender.toLowerCase() === "dog") {
            classes += " special-dog";
        } else if (message.sender.toLowerCase() === "cat") {
            classes += " special-cat";
        }

        const isFinalConclusion = message.msg_type === "summary" && (message.round_number === null || message.round_number === undefined);
        const isLongText = (message.content.length > 400 || message.msg_type === "summary" || message.turn_kind === "librarian_audit") && !isFinalConclusion;

        let contentHtml = "";
        if(isFinalConclusion) {
            contentHtml = '<div class="chat-content" style="font-size: 1.1em; font-weight: 500; border-left: 4px solid var(--warning); padding-left: 12px; margin-top: 8px;">' + linkCitations(message.content) + '</div>';
        } else if(isLongText) {
            let previewText = esc(message.content.substring(0, 100)) + "...";
            if (message.msg_type === "summary") previewText = "Round Summary";
            if (message.turn_kind === "librarian_audit") previewText = "Librarian Audit Log";

            contentHtml = '<details id="msg-det-' + message.id + '"><summary>Expand: ' + previewText + '</summary><div class="chat-content">' + linkCitations(message.content) + '</div></details>';
        } else {
            contentHtml = '<div class="chat-content">' + linkCitations(message.content) + '</div>';
        }

        const metaLabels = [];
        if (message.round_number !== null && message.round_number !== undefined) metaLabels.push('R' + message.round_number);
        if (message.turn_kind && message.turn_kind !== "base") metaLabels.push(message.turn_kind);
        if (message.msg_type !== "standard") metaLabels.push(message.msg_type);

        html += '<div class="' + classes + '">' +
          '<div class="chat-header">' +
          '<span class="chat-sender"><span style="color:var(--muted); font-weight:normal; margin-right:6px;">#' + trueIndex + '</span>' + esc(message.sender) + '</span>' +
          '<span class="chat-meta">' + (timeStr ? timeStr + ' | ' : '') + esc(metaLabels.join(' | ')) + '</span>' +
          '</div>' +
          contentHtml +
          '</div>';
      });
      
      if (currentRound !== null && !renderedVoteRounds.has(currentRound)) {
          html += renderVotesForRound(currentRound);
          renderedVoteRounds.add(currentRound);
      }
      
      if (snapshot.votes) {
          const admissionVotes = snapshot.votes.filter(v => v.vote_kind === 'candidate_admission');
          if (admissionVotes.length > 0) {
              let vHtml = '<div class="chat-bubble system" style="border-color:#bda2f7;">';
              vHtml += '<div class="chat-header"><span class="chat-sender" style="color:#bda2f7">ADMISSION VOTES</span><span class="chat-meta">Pre-workflow</span></div>';
              vHtml += '<details id="vote-det-admin"><summary>View ' + admissionVotes.length + ' Admission Votes</summary><div class="chat-content" style="font-family: var(--mono); font-size:12px;">';
              
              admissionVotes.forEach(v => {
                  const symbol = v.decision === 'yes' ? '✅' : (v.decision === 'no' ? '🛑' : '⚠️');
                  vHtml += '<strong style="color:#fff">' + esc(v.voter) + '</strong> (' + esc(v.subject) + '): ' + symbol + ' ' + esc(v.decision) + '<br>';
                  vHtml += '<span style="color:#aaa">Reason: ' + esc(v.reason) + '</span><br><br>';
              });
              vHtml += '</div></details></div>';
              html = vHtml + html;
          }
      }

      node.innerHTML = html;
    }

    function renderMSE(snapshot) {
      const node = document.getElementById('tab-mse');
      const review = snapshot.mse_review;
      if (!review) {
        renderEmpty(node, 'No MSE review data for this topic.');
        return;
      }
      const counts = review.review_counts || {};
      let html = '<div class="card" style="border-left:3px solid var(--accent);">';
      html += '<div class="card-title">MSE Review Snapshot</div>';
      if (snapshot.topic && snapshot.topic.id) {
        const mseJson = '/api/topic/' + snapshot.topic.id + '/mse_report';
        const mseMd = '/api/topic/' + snapshot.topic.id + '/mse_report/markdown';
        html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">' +
          '<a href="' + esc(mseJson) + '" target="_blank" rel="noopener" style="color:var(--accent);font-size:11px;text-decoration:none;border:1px solid var(--accent);padding:4px 8px;border-radius:4px;">Provenance JSON</a>' +
          '<a href="' + esc(mseMd) + '" target="_blank" rel="noopener" style="color:var(--accent);font-size:11px;text-decoration:none;border:1px solid var(--accent);padding:4px 8px;border-radius:4px;">Provenance MD</a>' +
          '</div>';
      }
      html += '<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;font-family:var(--mono);font-size:11px;color:var(--muted);">';
      ['documents','problems','components','artifacts','solver_runs','open_diagnostics','pending_components'].forEach(k => {
        html += '<div><span class="label">' + esc(k) + '</span>' + esc(counts[k] || 0) + '</div>';
      });
      html += '</div></div>';

      const docs = review.documents || [];
      html += '<div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;margin-bottom:4px;">Corpus Documents (' + docs.length + ')</div>';
      if (!docs.length) {
        html += '<div class="empty">No corpus documents indexed.</div>';
      } else {
        html += docs.map(doc => {
          const chunks = doc.chunks || [];
          let body = '<div><span class="label">Doc #' + esc(doc.id) + '</span><span class="label">' + esc(doc.doc_type || 'document') + '</span><span class="label">' + esc(doc.index_status || 'unknown') + '</span></div>';
          body += '<div class="card-title">' + esc(doc.title || doc.source_path || 'Untitled document') + '</div>';
          body += '<div style="font-size:11px;color:var(--muted);margin-bottom:6px;">chunks: ' + esc(doc.chunk_count || chunks.length) + ' | checksum: ' + esc((doc.checksum || '').substring(0, 12)) + '</div>';
          if (chunks.length) {
            body += '<details id="mse-doc-' + doc.id + '"><summary>View indexed chunks</summary>';
            body += chunks.map(ch => '<div style="border-top:1px solid var(--line);padding-top:6px;margin-top:6px;"><span class="label">[D' + esc(ch.id) + ']</span><span class="label">' + esc(ch.section_path || 'root') + '</span><div class="card-content">' + esc((ch.text || ch.table_markdown || '').substring(0, 500)) + '</div></div>').join('');
            body += '</details>';
          }
          return '<div class="card">' + body + '</div>';
        }).join('');
      }

      const problems = review.problems || [];
      html += '<div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;margin-top:16px;margin-bottom:4px;">Optimization Problems (' + problems.length + ')</div>';
      if (!problems.length) {
        html += '<div class="empty">No optimization problems captured.</div>';
      } else {
        html += problems.map(problem => {
          let body = '<div><span class="label">Problem #' + esc(problem.id) + '</span><span class="label">' + esc(problem.problem_class || 'unspecified') + '</span></div>';
          body += '<div class="card-title">' + esc(problem.title || 'Untitled problem') + '</div>';
          if (problem.source_text) body += '<div class="card-content">' + esc(problem.source_text.substring(0, 500)) + '</div>';

          const components = problem.components || [];
          body += '<details id="mse-comp-' + problem.id + '" open><summary>Components (' + components.length + ')</summary>';
          if (!components.length) {
            body += '<div class="empty">No components.</div>';
          } else {
            body += components.map(c => {
              let actions = '';
              if (c.review_status === 'candidate') {
                actions = '<button onclick="reviewComponent(' + c.id + ', \'reviewed\')" style="margin-right:6px;background:var(--accent);color:#000;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;font-size:11px;">Approve</button>' +
                  '<button onclick="reviewComponent(' + c.id + ', \'rejected\')" style="background:#d33c46;color:#fff;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;font-size:11px;">Reject</button>';
              }
              return '<div style="border-top:1px solid var(--line);padding-top:8px;margin-top:8px;">' +
                '<div><span class="label">' + esc(c.component_type) + '</span><span class="label">' + esc(c.review_status) + '</span><span class="label">' + esc(c.symbol || '') + '</span></div>' +
                '<div class="card-content">' + esc(c.natural_text || '') + '</div>' +
                (c.formal_text ? '<div style="font-family:var(--mono);font-size:11px;color:var(--fact);margin-top:4px;">' + esc(c.formal_text) + '</div>' : '') +
                (c.validation_notes ? '<div style="color:var(--warning);font-size:11px;margin-top:4px;">' + esc(c.validation_notes) + '</div>' : '') +
                (actions ? '<div style="margin-top:6px;">' + actions + '</div>' : '') +
                '</div>';
            }).join('');
          }
          body += '</details>';

          const artifacts = problem.artifacts || [];
          body += '<details id="mse-art-' + problem.id + '"><summary>Artifacts (' + artifacts.length + ')</summary>';
          if (!artifacts.length) {
            body += '<div class="empty">No artifacts.</div>';
          } else {
            body += artifacts.map(a => '<div style="border-top:1px solid var(--line);padding-top:8px;margin-top:8px;"><span class="label">O' + esc(a.id) + '</span><span class="label">' + esc(a.model_language) + '</span><span class="label">' + esc(a.parser_status) + '</span>' + (a.repair_status ? '<span class="label">' + esc(a.repair_status) + '</span>' : '') + '<pre style="background:#1a1c20;padding:8px;border-radius:4px;overflow-x:auto;font-size:11px;max-height:220px;">' + esc(a.content || '') + '</pre></div>').join('');
          }
          body += '</details>';

          const runs = problem.solver_runs || [];
          body += '<details id="mse-run-' + problem.id + '"><summary>Solver Runs (' + runs.length + ')</summary>';
          if (!runs.length) {
            body += '<div class="empty">No solver runs.</div>';
          } else {
            body += runs.map(r => '<div style="border-top:1px solid var(--line);padding-top:8px;margin-top:8px;"><span class="label">Run #' + esc(r.id) + '</span><span class="label">' + esc(r.solver_backend) + '</span><span class="label">' + esc(r.status) + '</span><div style="font-family:var(--mono);font-size:11px;color:var(--muted);">objective: ' + esc(r.objective_value ?? '') + ' | evidence: ' + (r.code_evidence_id ? '[E' + esc(r.code_evidence_id) + ']' : '-') + '</div>' + (r.stderr ? '<pre style="background:#2a1c1c;padding:8px;border-radius:4px;font-size:11px;max-height:160px;overflow:auto;">' + esc(r.stderr) + '</pre>' : '') + '</div>').join('');
          }
          body += '</details>';

          const diagnostics = problem.diagnostics || [];
          body += '<details id="mse-diag-' + problem.id + '" open><summary>Diagnostics (' + diagnostics.length + ')</summary>';
          if (!diagnostics.length) {
            body += '<div class="empty">No diagnostics.</div>';
          } else {
            body += diagnostics.map(d => {
              const action = d.status === 'open' ? '<button onclick="resolveDiagnostic(' + d.id + ', \'resolved\')" style="background:var(--warning);color:#000;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;font-size:11px;margin-top:6px;">Resolve</button>' : '';
              return '<div style="border-top:1px solid var(--line);padding-top:8px;margin-top:8px;">' +
                '<div><span class="label">Diag #' + esc(d.id) + '</span><span class="label">' + esc(d.diagnostic_type) + '</span><span class="label">' + esc(d.severity) + '</span><span class="label">' + esc(d.status) + '</span></div>' +
                '<div class="card-content">' + esc(d.message || '') + '</div>' + action + '</div>';
            }).join('');
          }
          body += '</details>';
          return '<div class="card" style="border-left:3px solid #60a5fa;">' + body + '</div>';
        }).join('');
      }

      node.innerHTML = html;
    }

    function renderKB(snapshot) {
      renderMSE(snapshot);
      // 1. Facts
      const factsNode = document.getElementById('tab-facts');
      if (!snapshot.facts || !snapshot.facts.length) {
        renderEmpty(factsNode, 'No accepted facts yet.');
      } else {
        factsNode.innerHTML = snapshot.facts.map(f => {
          const key = 'F' + f.id;
          const superseded = f.superseded_by != null;
          const cardCls = superseded ? 'card fact-superseded' : 'card';
          let header = '<span class="label">[' + key + ']</span>';
          header += '<span class="label" style="color:var(--fact)">' + esc(f.review_status) + '</span>';
          header += _sourceKindSpan(f.source_kind);
          if (f.confidence_score != null) header += '<span class="label" style="color:var(--muted);margin-left:4px;">conf: ' + Number(f.confidence_score).toFixed(2) + '</span>';
          if (superseded) header += '<span class="label" style="color:#d33c46;margin-left:4px;">superseded by F' + f.superseded_by + '</span>';
          const edgeCount = (EDGE_INDEX[key] || []).length;
          if (edgeCount) header += '<span style="color:var(--muted);font-size:10px;margin-left:6px;">' + edgeCount + ' edge(s)</span>';

          let body = '<div class="fact-content" style="font-weight:bold; margin-bottom:8px;">' + linkCitations(f.summary || f.content) + '</div>';
          if (f.summary) body += '<details id="f-det-' + f.id + '"><summary>View Full Text</summary><div class="card-content" style="margin-top:8px; font-size:12px; border-top:1px solid var(--line); padding-top:8px;">' + linkCitations(f.content) + '</div></details>';
          body += _renderEdgeDetails(key, 'f-edges-' + f.id);
          return '<div class="' + cardCls + '" id="fact-' + f.id + '"><div>' + header + '</div>' + body + '</div>';
        }).join('');
      }
      
      // 2. Claims
      const claimsNode = document.getElementById('tab-claims');
      if (!snapshot.claims || !snapshot.claims.length) {
        renderEmpty(claimsNode, 'No claims generated yet.');
      } else {
        claimsNode.innerHTML = snapshot.claims.map(c => {
          const key = 'C' + c.id;
          const statusColor = c.status === 'active' ? 'var(--fact)' : (c.status === 'contested' ? '#f59e0b' : 'var(--muted)');
          let supportHtml = '';
          if (c.support_fact_ids_json) {
            try {
              const fids = JSON.parse(c.support_fact_ids_json);
              if (Array.isArray(fids) && fids.length) {
                supportHtml = '<div style="margin-top:4px;font-size:11px;color:var(--muted);">Supporting: ' + fids.map(fid => '<span class="citation" style="cursor:pointer;" onmouseover="showTooltip(event, \'F' + fid + '\')" onmouseout="hideTooltip()" onclick="jumpToFact(' + fid + ')">[F' + fid + ']</span>').join(' ') + '</div>';
              }
            } catch(e) {}
          }
          const scoreHtml = c.claim_score != null ? '<span class="label" style="color:var(--muted);margin-left:4px;">score: ' + Number(c.claim_score).toFixed(2) + '</span>' : '';
          const rationaleHtml = c.rationale_short ? '<details id="c-rat-' + c.id + '"><summary style="font-size:11px;color:var(--muted);cursor:pointer;">Rationale</summary><div style="margin-top:4px;font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:4px;">' + esc(c.rationale_short) + '</div></details>' : '';
          const edgeHtml = _renderEdgeDetails(key, 'c-edges-' + c.id);
          return '<div class="card">' +
            '<div><span class="label">[' + key + ']</span><span class="label" style="color:' + statusColor + '">' + esc(c.status || 'unknown') + '</span>' + scoreHtml + '</div>' +
            '<div class="fact-content" style="font-weight:bold; margin-bottom:8px;">' + linkCitations(c.summary || c.content) + '</div>' +
            supportHtml +
            rationaleHtml +
            (c.summary ? '<details id="c-det-' + c.id + '"><summary>View Full Text</summary><div class="card-content" style="margin-top:8px; font-size:12px; border-top:1px solid var(--line); padding-top:8px;">' + linkCitations(c.content) + '</div></details>' : '') +
            edgeHtml +
            '</div>';
        }).join('');
      }

      // 3. Ledger
      const ledgerNode = document.getElementById('tab-ledger');
      if (!snapshot.ledger_entries || !snapshot.ledger_entries.length) {
        renderEmpty(ledgerNode, 'No ledger entries yet.');
      } else {
        let ledgerHtml = '';

        // Build contested IDs set
        const contestedIds = new Set();
        if (snapshot.contested_pairs) {
          snapshot.contested_pairs.forEach(p => {
            if (p.entries) p.entries.forEach(e => contestedIds.add(e.id));
          });
        }

        // Build edge map: entry_id -> descriptions
        const edgeMap = {};
        if (snapshot.ledger_edges) {
          snapshot.ledger_edges.forEach(edge => {
            const desc = edge.edge_type + (edge.created_by ? ' (' + edge.created_by + ')' : '');
            [edge.from_entry_id, edge.to_entry_id].forEach(eid => {
              if (!edgeMap[eid]) edgeMap[eid] = [];
              edgeMap[eid].push(desc);
            });
          });
        }

        // Contested summary card
        if (snapshot.contested_pairs && snapshot.contested_pairs.length > 0) {
          ledgerHtml += '<div class="card" style="border-left: 3px solid var(--warning); background: rgba(255,180,0,0.05);">';
          ledgerHtml += '<div class="card-title" style="color:var(--warning);">Contested Data (' + snapshot.contested_pairs.length + ' conflicts)</div>';
          ledgerHtml += '<div class="card-content" style="font-size:12px;">';
          snapshot.contested_pairs.forEach(p => {
            if (!p.entries || p.entries.length < 2) return;
            const ids = p.entries.map(e => 'L' + e.id).join(' vs ');
            const values = p.entries.map(e => (e.source_ref || '?') + ' says ' + (e.value || '?')).join(', ');
            ledgerHtml += '<div style="margin-bottom:4px;">' + esc(ids) + ': <strong>' + esc(p.entity_name) + '</strong> ' + esc(p.attribute_name) + ' for ' + esc(p.timeframe) + ' — ' + esc(values) + '</div>';
          });
          ledgerHtml += '</div></div>';
        }

        // Group entries by entity
        const grouped = {};
        snapshot.ledger_entries.forEach(e => {
          const name = e.entity_name || 'Unknown';
          if (!grouped[name]) grouped[name] = [];
          grouped[name].push(e);
        });

        // Render per-entity cards
        Object.keys(grouped).forEach(entityName => {
          const entries = grouped[entityName];
          ledgerHtml += '<div class="card">';
          ledgerHtml += '<div class="card-title">' + esc(entityName) + ' (' + entries.length + ' entries)</div>';
          ledgerHtml += '<div style="overflow-x:auto;"><table style="width:100%; border-collapse:collapse; font-size:12px; font-family:var(--mono);">';
          ledgerHtml += '<tr style="border-bottom:1px solid var(--line); color:var(--muted);">';
          ledgerHtml += '<th style="text-align:left;padding:4px;">ID</th><th style="text-align:left;padding:4px;">Attribute</th><th style="text-align:left;padding:4px;">Value</th><th style="text-align:left;padding:4px;">Time</th><th style="text-align:left;padding:4px;">Source</th><th style="text-align:left;padding:4px;">Status</th></tr>';
          entries.forEach(e => {
            const isContested = contestedIds.has(e.id);
            const rowStyle = isContested ? 'background:rgba(255,180,0,0.1);' : '';
            let val = esc(e.value || '');
            if (e.unit) val += ' ' + esc(e.unit);
            const edges = edgeMap[e.id];
            const edgeBadge = edges ? ' <span style="background:var(--line);padding:1px 4px;border-radius:2px;font-size:10px;cursor:help;" title="' + esc(edges.join(', ')) + '">[' + edges.length + ' edges]</span>' : '';
            ledgerHtml += '<tr style="border-bottom:1px solid var(--line);' + rowStyle + '">';
            ledgerHtml += '<td style="padding:4px;color:var(--accent);">L' + e.id + edgeBadge + '</td>';
            ledgerHtml += '<td style="padding:4px;">' + esc(e.attribute_name) + '</td>';
            ledgerHtml += '<td style="padding:4px;color:#fff;font-weight:bold;">' + val + '</td>';
            ledgerHtml += '<td style="padding:4px;color:var(--muted);">' + esc(e.normalized_timeframe || '') + '</td>';
            ledgerHtml += '<td style="padding:4px;">' + esc(e.source_ref || '') + '</td>';
            ledgerHtml += '<td style="padding:4px;">' + esc(e.status || '') + '</td>';
            ledgerHtml += '</tr>';
          });
          ledgerHtml += '</table></div></div>';
        });

        ledgerNode.innerHTML = ledgerHtml;
      }

      // 4. Candidates (Pending tab — 3 sub-sections)
      const candsNode = document.getElementById('tab-cands');
      let candsHtml = '';

      // 4a. Fact Candidates
      const fCands = snapshot.fact_candidates || [];
      candsHtml += '<div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;margin-bottom:4px;">Fact Candidates (' + fCands.length + ')</div>';
      if (!fCands.length) {
        candsHtml += '<div class="empty">No fact candidates yet.</div>';
      } else {
        candsHtml += fCands.map(c => {
          let extra = '';
          if (c.status !== 'pending' && c.review_note) {
              extra = '<details id="cand-det-' + c.id + '" style="margin-top:8px; border-color:var(--warning)"><summary>Audit Reason</summary><div style="color:var(--warning); font-size:12px; margin-top:4px;">' + esc(c.review_note) + '</div></details>';
          }
          let labels = '<span class="label">Cand #' + c.id + '</span><span class="label" style="color:var(--pending)">' + esc(c.status) + '</span>';
          labels += _sourceKindSpan(c.source_kind);
          if (c.fact_stage && c.fact_stage !== 'synthesized') labels += '<span class="label" style="color:var(--muted)">' + esc(c.fact_stage) + '</span>';
          if (c.confidence_score != null) labels += '<span class="label" style="color:var(--muted)">conf: ' + Number(c.confidence_score).toFixed(2) + '</span>';
          return '<div class="card">' +
          '<div>' + labels + '</div>' +
          '<div class="candidate-content">' + linkCitations(c.candidate_text) + '</div>' +
          extra +
          '</div>';
        }).join('');
      }

      // 4b. Claim Candidates
      const cCands = snapshot.claim_candidates || [];
      candsHtml += '<div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;margin-top:16px;margin-bottom:4px;">Claim Candidates (' + cCands.length + ')</div>';
      if (!cCands.length) {
        candsHtml += '<div class="empty">No claim candidates yet.</div>';
      } else {
        candsHtml += cCands.map(c => {
          let extra = '';
          if (c.rationale_short) {
            extra += '<details id="ccand-rat-' + c.id + '" style="margin-top:8px;"><summary style="font-size:11px;color:var(--muted);cursor:pointer;">Rationale</summary><div style="font-size:12px;color:var(--muted);margin-top:4px;">' + esc(c.rationale_short) + '</div></details>';
          }
          if (c.status !== 'pending' && c.review_note) {
            extra += '<details id="ccand-det-' + c.id + '" style="margin-top:8px; border-color:var(--warning)"><summary>Audit Reason</summary><div style="color:var(--warning); font-size:12px; margin-top:4px;">' + esc(c.review_note) + '</div></details>';
          }
          return '<div class="card" style="border-left:3px solid #00bcd4;">' +
          '<div><span class="label">CCand #' + c.id + '</span><span class="label" style="color:#00bcd4">' + esc(c.status) + '</span></div>' +
          '<div class="candidate-content">' + esc(c.candidate_text) + '</div>' +
          extra +
          '</div>';
        }).join('');
      }

      // 4c. Ledger Pending
      const lPending = snapshot.ledger_pending || [];
      candsHtml += '<div style="font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;margin-top:16px;margin-bottom:4px;">Ledger Pending (' + lPending.length + ')</div>';
      if (!lPending.length) {
        candsHtml += '<div class="empty">No pending ledger entries.</div>';
      } else {
        candsHtml += lPending.map(p => {
          let ttl = p.ttl_expires_round ? 'TTL: round ' + p.ttl_expires_round : 'No TTL';
          let missing = p.missing_fields ? 'Missing: ' + p.missing_fields : '';
          return '<div class="card" style="border-left:3px solid var(--warning);">' +
          '<div><span class="label">LP #' + p.id + '</span><span class="label" style="color:var(--warning)">' + esc(ttl) + '</span></div>' +
          '<div class="candidate-content">' + esc(p.raw_text) + '</div>' +
          (missing ? '<div style="color:var(--warning);font-size:11px;margin-top:4px;font-family:var(--mono);">' + esc(missing) + '</div>' : '') +
          '</div>';
        }).join('');
      }

      candsNode.innerHTML = candsHtml;

      // 6. Web Evidence
      const webNode = document.getElementById('tab-web');
      if (!snapshot.web_evidence || !snapshot.web_evidence.length) {
        webNode.innerHTML = '<div class="empty-state">No web searches performed.</div>';
      } else {
        webNode.innerHTML = snapshot.web_evidence.map(w =>
          '<div class="card">' +
          '<div><span class="label">[W' + w.id + ']</span><a href="' + (/^https?:\/\//i.test(w.url) ? esc(w.url) : '#') + '" target="_blank" rel="noopener noreferrer" style="color:var(--accent);font-size:11px;">' + esc(w.source_domain) + '</a></div>' +
          '<div class="card-title">' + esc(w.title) + '</div>' +
          '<div class="card-content">' + esc(w.snippet) + '</div>' +
          '</div>'
        ).join('');
      }

      // 7. Code Evidence (own tab)
      const codeNode = document.getElementById('tab-code');
      if (!snapshot.code_evidence || !snapshot.code_evidence.length) {
        codeNode.innerHTML = '<div class="empty-state">No code experiments run yet.</div>';
      } else {
        codeNode.innerHTML = snapshot.code_evidence.map(e => {
          const status = e.success ? '<span style="color:#3cb878;">PASSED</span>' : '<span style="color:#d33c46;">FAILED</span>';
          const isGrid = e.parent_evidence_id && (e.hypothesis || '').startsWith('Grid sweep');
          const tierTag = isGrid ? ' <span style="color:#3cb878;font-size:11px;">GRID of [E' + e.parent_evidence_id + ']</span>'
            : e.parent_evidence_id ? ' <span style="color:#a78bfa;font-size:11px;">REVIEW of [E' + e.parent_evidence_id + ']</span>'
            : (e.hypothesis || '').startsWith('CALC:') ? ' <span style="color:#60a5fa;font-size:11px;">CALC</span>'
            : ' <span style="color:#f59e0b;font-size:11px;">VERIFY</span>';
          const reviewBadge = (!e.parent_evidence_id && e.review_count > 0) ? ' <span style="color:#94a3b8;font-size:10px;" title="Reviewed ' + e.review_count + ' time(s) without issues">[reviewed x' + e.review_count + ']</span>' : '';
          return '<div class="card">' +
            '<div><span class="label">[E' + e.id + ']</span> ' + status + tierTag + reviewBadge + ' <span style="color:var(--muted);font-size:11px;">(' + e.iterations + ' iter, ' + (e.execution_time_s || 0).toFixed(1) + 's, by ' + esc(e.requesting_role || '?') + ')</span></div>' +
            '<div class="card-title">' + esc(e.summary || e.hypothesis.split('\\n')[0]) + '</div>' +
            '<details id="e-det-' + e.id + '"><summary>View Code & Output</summary>' +
            '<pre style="background:#1a1c20;padding:8px;border-radius:4px;overflow-x:auto;font-size:11px;max-height:300px;">' + esc(e.source_code || '') + '</pre>' +
            (e.stdout ? '<div style="margin-top:8px;"><span class="label">stdout:</span><pre style="background:#1a1c20;padding:8px;border-radius:4px;font-size:11px;max-height:200px;overflow:auto;">' + esc(e.stdout) + '</pre></div>' : '') +
            (e.stderr ? '<div style="margin-top:8px;"><span class="label" style="color:#d33c46;">stderr:</span><pre style="background:#2a1c1c;padding:8px;border-radius:4px;font-size:11px;max-height:200px;overflow:auto;">' + esc(e.stderr) + '</pre></div>' : '') +
            '</details></div>';
        }).join('');
      }
    }

    async function refresh() {
      // Skip refresh if user has text selected (preserve selection)
      const sel = window.getSelection();
      if (sel && sel.toString().trim().length > 0) return;

      let snapshot;

      if (window.ORBIT_STATIC_DATA) {
          // Static Mode
          let subId = currentSubtopicId;
          if (!subId && window.ORBIT_STATIC_DATA.subtopics.length > 0) {
              subId = window.ORBIT_STATIC_DATA.subtopics[window.ORBIT_STATIC_DATA.subtopics.length - 1].id;
          }
          if (subId && window.ORBIT_STATIC_DATA.subtopic_data[subId]) {
              snapshot = window.ORBIT_STATIC_DATA.subtopic_data[subId];
          } else {
              snapshot = window.ORBIT_STATIC_DATA.subtopic_data["default"] || window.ORBIT_STATIC_DATA;
          }
      } else {
          // Dynamic Mode
          try {
            let url = '/api/dashboard';
            const params = new URLSearchParams();
            if (currentTopicId) params.set('topic_id', currentTopicId);
            if (currentSubtopicId) params.set('subtopic_id', currentSubtopicId);
            const qs = params.toString();
            if (qs) url += '?' + qs;
            const response = await fetch(url, { cache: 'no-store' });
            if (!response.ok) { console.warn('Dashboard fetch failed:', response.status); return; }
            snapshot = await response.json();
          } catch (err) {
            console.warn('Dashboard refresh error:', err);
            return;
          }
      }
      
      const detailsState = {};
      document.querySelectorAll('details').forEach(el => {
         if(el.id) detailsState[el.id] = el.open;
      });
      
      // Remember scroll position of timeline
      const timelineScroll = document.getElementById('scroll-timeline').scrollTop;

      updateKnowledgeMap(snapshot);
      buildEdgeIndex(snapshot);
      await loadTopicOptions(snapshot.topic ? snapshot.topic.id : null);
      renderTopic(snapshot);
      renderPlan(snapshot);
      renderMessages(snapshot);
      renderKB(snapshot);

      // Show report link if report exists
      const reportLink = document.getElementById('report-link');
      if (snapshot.topic && snapshot.topic.report_json) {
        reportLink.href = '/api/topic/' + snapshot.topic.id + '/report/html';
        reportLink.style.display = 'inline-block';
      }

      document.querySelectorAll('details').forEach(el => {
         if(el.id && detailsState[el.id]) el.open = true;
      });
      document.getElementById('scroll-timeline').scrollTop = timelineScroll;
      
      if (window.ORBIT_STATIC_DATA || (snapshot.topic && snapshot.topic.status === 'Closed')) {
          if (window.refreshInterval) clearInterval(window.refreshInterval);
      }
    }

    refresh();
    window.refreshInterval = setInterval(() => { if(isAutoRefresh) refresh(); }, 3000);
  </script>
</body>
</html>"""


async def index(request):
    return web.Response(text=render_dashboard_html(), content_type="text/html")


async def dashboard(request):
    raw_topic_id = request.query.get("topic_id")
    raw_id = request.query.get("subtopic_id")
    topic_id = None
    if raw_topic_id:
        try:
            topic_id = str(int(raw_topic_id))
        except (ValueError, TypeError):
            return web.json_response({"error": "Invalid topic_id"}, status=400)
    subtopic_id = None
    if raw_id:
        try:
            subtopic_id = str(int(raw_id))
        except (ValueError, TypeError):
            return web.json_response({"error": "Invalid subtopic_id"}, status=400)
    snapshot = await asyncio.to_thread(
        partial(build_dashboard_snapshot, topic_id=topic_id, subtopic_id=subtopic_id)
    )
    return web.json_response(snapshot)


async def handle_mse_review(request):
    try:
        topic_id = int(request.match_info["id"])
    except (TypeError, ValueError):
        return web.json_response({"error": "Invalid topic id"}, status=400)
    if not api.get_topic(topic_id):
        return web.json_response({"error": "Topic not found"}, status=404)
    loop = asyncio.get_running_loop()
    snapshot = await loop.run_in_executor(
        None, partial(api.get_mse_review_snapshot, topic_id)
    )
    return web.json_response(snapshot)


async def handle_mse_provenance_report(request):
    try:
        topic_id = int(request.match_info["id"])
    except (TypeError, ValueError):
        return web.json_response({"error": "Invalid topic id"}, status=400)
    if not api.get_topic(topic_id):
        return web.json_response({"error": "Topic not found"}, status=404)
    report = await asyncio.to_thread(api.get_mse_provenance_report, topic_id)
    return web.json_response(report)


async def handle_mse_provenance_markdown(request):
    try:
        topic_id = int(request.match_info["id"])
    except (TypeError, ValueError):
        return web.json_response({"error": "Invalid topic id"}, status=400)
    if not api.get_topic(topic_id):
        return web.json_response({"error": "Topic not found"}, status=404)
    report = await asyncio.to_thread(api.get_mse_provenance_report, topic_id)
    markdown = api.render_mse_provenance_markdown(report)
    return web.Response(text=markdown, content_type="text/markdown")


async def handle_ingest_corpus_document(request):
    try:
        topic_id = int(request.match_info["id"])
    except (TypeError, ValueError):
        return web.json_response({"error": "Invalid topic id"}, status=400)
    if not api.get_topic(topic_id):
        return web.json_response({"error": "Topic not found"}, status=404)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    title = str(body.get("title") or "").strip()
    text = str(body.get("text") or "").strip()
    if not title:
        return web.json_response({"error": "title is required"}, status=400)
    if not text:
        return web.json_response({"error": "text is required"}, status=400)
    if len(text) > 2_000_000:
        return web.json_response({"error": "text is too large"}, status=413)

    kwargs = {
        "topic_id": topic_id,
        "title": title,
        "text": text,
        "doc_type": str(body.get("doc_type") or "text").strip() or "text",
        "author": body.get("author"),
        "source_path": body.get("source_path"),
        "source_url": body.get("source_url"),
        "access_scope": str(body.get("access_scope") or "topic").strip() or "topic",
        "metadata": body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
    }
    embed = bool(body.get("embed", False))
    from . import corpus

    if embed:
        result = await corpus.aingest_text_document(embed=True, **kwargs)
    else:
        result = await asyncio.to_thread(corpus.ingest_text_document, **kwargs)
    document = await asyncio.to_thread(api.get_corpus_document, result["document_id"])
    return web.json_response({"ok": True, **result, "document": document})


async def handle_review_mse_component(request):
    try:
        component_id = int(request.match_info["id"])
    except (KeyError, TypeError, ValueError):
        return web.json_response({"error": "Invalid component id"}, status=400)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    status = str(body.get("review_status") or "").strip()
    allowed = {"candidate", "reviewed", "rejected", "formalized", "executable"}
    if status not in allowed:
        return web.json_response({"error": "Invalid review_status"}, status=400)
    note = body.get("validation_notes")
    ok = await asyncio.to_thread(
        api.update_optimization_component_review,
        component_id,
        review_status=status,
        validation_notes=str(note) if note is not None else None,
    )
    if not ok:
        return web.json_response({"error": "Component not found"}, status=404)
    return web.json_response({"ok": True, "component_id": component_id})


async def handle_update_mse_diagnostic(request):
    try:
        diagnostic_id = int(request.match_info["id"])
    except (KeyError, TypeError, ValueError):
        return web.json_response({"error": "Invalid diagnostic id"}, status=400)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    status = str(body.get("status") or "").strip()
    allowed = {"open", "resolved", "dismissed", "closed"}
    if status not in allowed:
        return web.json_response({"error": "Invalid status"}, status=400)
    resolution = body.get("resolution")
    ok = await asyncio.to_thread(
        api.update_model_diagnostic_status,
        diagnostic_id,
        status=status,
        resolution=str(resolution) if resolution is not None else None,
    )
    if not ok:
        return web.json_response({"error": "Diagnostic not found"}, status=404)
    return web.json_response({"ok": True, "diagnostic_id": diagnostic_id})


async def health(request):
    topic = await asyncio.to_thread(api.get_current_topic)
    subtopic = None
    if topic:
        subtopic = await asyncio.to_thread(api.get_open_subtopic, topic["id"])
    return web.json_response(
        {
            "ok": True,
            "topic_id": topic["id"] if topic else None,
            "current_subtopic_id": subtopic["id"] if subtopic else None,
        }
    )


async def handle_list_topics(request):
    topics = await asyncio.to_thread(api.list_topics)
    current = await asyncio.to_thread(api.get_current_topic)
    return web.json_response(
        {
            "topics": topics,
            "current_topic_id": current["id"] if current else None,
        }
    )


# ---------------------------------------------------------------------------
# Phase F.1: Topic creation + config endpoints
# ---------------------------------------------------------------------------


def _int_param(request, key: str) -> int:
    """Parse an integer from a URL path parameter. Raises HTTPBadRequest on invalid."""
    try:
        return int(request.match_info[key])
    except (ValueError, KeyError):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": f"Invalid {key}"}),
            content_type="application/json",
        )


def _str_field(body: dict, key: str) -> str:
    """Safely extract a string field from a JSON body, coercing None/non-str to ''."""
    val = body.get(key)
    return str(val).strip() if isinstance(val, str) else ""


async def handle_create_topic(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Body must be a JSON object"}, status=400)
    summary = _str_field(body, "summary")
    detail = _str_field(body, "detail")
    if not summary:
        return web.json_response({"error": "summary is required"}, status=400)
    config = body.get("config")
    try:
        topic_id = await asyncio.to_thread(api.create_topic, summary, detail, config)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"topic_id": topic_id}, status=201)


async def handle_get_topic_config(request):
    topic_id = _int_param(request, "id")
    from . import topic_config

    cfg = await asyncio.to_thread(topic_config.get_all, topic_id)
    return web.json_response(cfg)


async def handle_put_topic_config(request):
    topic_id = _int_param(request, "id")
    topic = await asyncio.to_thread(api.get_topic, topic_id)
    if not topic:
        return web.json_response({"error": "Topic not found"}, status=404)
    if topic["status"] not in ("Paused", "Queued", "Started"):
        return web.json_response(
            {"error": "Config can only be updated when Paused/Queued/Started"},
            status=409,
        )
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Body must be a JSON object"}, status=400)
    from . import topic_config

    ok, err = await asyncio.to_thread(topic_config.set_bulk, topic_id, body)
    if not ok:
        return web.json_response({"error": err}, status=400)
    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# Phase F.2: HITL endpoints
# ---------------------------------------------------------------------------


async def handle_resume_topic(request):
    topic_id = _int_param(request, "id")
    topic = await asyncio.to_thread(api.get_topic, topic_id)
    if not topic:
        return web.json_response({"error": "Topic not found"}, status=404)
    if topic["status"] != "Paused":
        return web.json_response({"error": "Topic is not paused"}, status=409)
    await asyncio.to_thread(api.resume_topic, topic_id)
    return web.json_response({"ok": True})


async def handle_inject_knowledge(request):
    topic_id = _int_param(request, "id")
    topic = await asyncio.to_thread(api.get_topic, topic_id)
    if not topic:
        return web.json_response({"error": "Topic not found"}, status=404)
    if topic.get("status") == "Closed":
        return web.json_response(
            {"error": "Cannot inject into a closed topic"}, status=400
        )
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Body must be a JSON object"}, status=400)
    injection_type = _str_field(body, "type")
    content = _str_field(body, "content")
    if injection_type not in ("url", "text", "search_query"):
        return web.json_response(
            {"error": "type must be url, text, or search_query"}, status=400
        )
    if not content:
        return web.json_response({"error": "content is required"}, status=400)
    subtopic_id = body.get("subtopic_id")
    if subtopic_id is not None:
        try:
            subtopic_id = int(subtopic_id)
        except (ValueError, TypeError):
            return web.json_response(
                {"error": "subtopic_id must be an integer"}, status=400
            )
    inj_id = await asyncio.to_thread(
        api.inject_knowledge, topic_id, injection_type, content, subtopic_id
    )
    analytics.capture(
        f"topic_{topic_id}",
        "knowledge_injected",
        {"injection_type": injection_type, "content_length": len(content)},
    )
    return web.json_response({"injection_id": inj_id}, status=201)


async def handle_pause_status(request):
    topic_id = _int_param(request, "id")
    topic = await asyncio.to_thread(api.get_topic, topic_id)
    if not topic:
        return web.json_response({"error": "Topic not found"}, status=404)
    injections = await asyncio.to_thread(api.get_pending_injections, topic_id)
    return web.json_response(
        {
            "status": topic["status"],
            "paused_at_stage": topic.get("paused_at_stage"),
            "pending_injections": injections,
        }
    )


# ---------------------------------------------------------------------------
# Phase F.3: Report endpoints
# ---------------------------------------------------------------------------


_report_tasks: dict[int, asyncio.Task] = {}


def _on_report_done(topic_id: int, task: asyncio.Task) -> None:
    _report_tasks.pop(topic_id, None)
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        import logging

        logging.getLogger(__name__).warning(
            "[report] Background report failed for topic %d: %s", topic_id, exc
        )


async def handle_trigger_report(request):
    topic_id = _int_param(request, "id")
    topic = await asyncio.to_thread(api.get_topic, topic_id)
    if not topic:
        return web.json_response({"error": "Topic not found"}, status=404)
    # Prevent duplicate concurrent generation
    existing = _report_tasks.get(topic_id)
    if existing and not existing.done():
        return web.json_response(
            {"error": "Report generation already in progress"}, status=409
        )
    from . import report

    task = asyncio.ensure_future(report.generate_report(topic_id))
    task.add_done_callback(lambda t: _on_report_done(topic_id, t))
    _report_tasks[topic_id] = task
    return web.json_response({"ok": True, "status": "generating"}, status=202)


async def handle_get_report_json(request):
    topic_id = _int_param(request, "id")
    raw = await asyncio.to_thread(api.get_report, topic_id)
    if not raw:
        return web.json_response({"error": "No report"}, status=404)
    # Return raw JSON string directly to avoid redundant parse+serialize
    return web.Response(text=raw, content_type="application/json")


def _parse_report(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


async def handle_get_report_html(request):
    topic_id = _int_param(request, "id")
    raw = await asyncio.to_thread(api.get_report, topic_id)
    if not raw:
        return web.json_response({"error": "No report"}, status=404)
    report_data = _parse_report(raw)
    if not report_data:
        return web.json_response({"error": "Corrupt report data"}, status=500)
    from . import report as report_mod

    html = report_mod.render_html_report(report_data)
    return web.Response(text=html, content_type="text/html")


async def handle_get_report_markdown(request):
    topic_id = _int_param(request, "id")
    raw = await asyncio.to_thread(api.get_report, topic_id)
    if not raw:
        return web.json_response({"error": "No report"}, status=404)
    report_data = _parse_report(raw)
    if not report_data:
        return web.json_response({"error": "Corrupt report data"}, status=500)
    from . import report as report_mod

    md = report_mod.render_markdown_report(report_data)
    return web.Response(text=md, content_type="text/markdown")


# ---------------------------------------------------------------------------
# Phase F.4: Knowledge Graph
# ---------------------------------------------------------------------------


def build_graph_data(topic_id: int) -> dict:
    """Build nodes + edges for knowledge graph visualization."""
    nodes = []
    edges = []
    facts = api.get_facts(topic_id, limit=500)
    claims = api.get_claims(topic_id, limit=500)
    web_evidence = api.get_web_evidence_for_topic(topic_id)
    code_evidence = api.get_code_evidence_for_topic(topic_id)
    ledger_entries = api.get_ledger_entries_with_names(topic_id)
    knowledge_edges = api.get_knowledge_edges(topic_id)

    for f in facts:
        full = f.get("summary") or f["content"]
        nodes.append(
            {
                "id": f"F{f['id']}",
                "type": "fact",
                "label": full[:80],
                "full_label": full,
                "status": f.get("review_status") or "accepted",
            }
        )
    for c in claims:
        full = c.get("summary") or c["content"]
        nodes.append(
            {
                "id": f"C{c['id']}",
                "type": "claim",
                "label": full[:80],
                "full_label": full,
                "status": c.get("status", "active"),
            }
        )
    for w in web_evidence:
        full = w.get("title") or w.get("snippet", "")
        nodes.append(
            {
                "id": f"W{w['id']}",
                "type": "web",
                "label": full[:80],
                "full_label": full,
                "status": "verified" if w.get("verified") else "unverified",
            }
        )
    for ce in code_evidence:
        full = ce.get("summary") or ce["hypothesis"]
        nodes.append(
            {
                "id": f"E{ce['id']}",
                "type": "code",
                "label": full[:80],
                "full_label": full,
                "status": "success" if ce.get("success") else "failed",
            }
        )
    for le in ledger_entries:
        full = f"{le.get('entity_name', '?')}.{le.get('attribute_name', '?')}={le.get('value', '?')}"
        nodes.append(
            {
                "id": f"L{le['id']}",
                "type": "ledger",
                "label": full[:80],
                "full_label": full,
                "status": le.get("status", "accepted"),
            }
        )

    type_prefix_map = {
        "fact": "F",
        "claim": "C",
        "web_evidence": "W",
        "code_evidence": "E",
        "ledger": "L",
    }
    for ke in knowledge_edges:
        src_prefix = type_prefix_map.get(ke["source_type"], "")
        tgt_prefix = type_prefix_map.get(ke["target_type"], "")
        if src_prefix and tgt_prefix:
            edges.append(
                {
                    "source": f"{src_prefix}{ke['source_id']}",
                    "target": f"{tgt_prefix}{ke['target_id']}",
                    "relation": ke["relation"],
                    "confidence": ke.get("confidence"),
                }
            )

    # Add derived_from edges for code evidence parent chains (review + grid)
    for ce in code_evidence:
        parent_id = ce.get("parent_evidence_id")
        if parent_id:
            edges.append(
                {
                    "source": f"E{ce['id']}",
                    "target": f"E{parent_id}",
                    "relation": "derived_from",
                }
            )

    return {"nodes": nodes, "edges": edges}


def render_graph_html() -> str:
    """Render a standalone D3.js knowledge graph page."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ORBIT Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root { --bg: #0d1117; --fg: #e6edf3; --accent: #58a6ff; --card-bg: #161b22; --border: #30363d; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--fg); overflow: hidden; }
  #graph-container { width: 100vw; height: 100vh; }
  svg { width: 100%; height: 100%; }
  .sidebar { position: fixed; top: 0; right: -350px; width: 350px; height: 100vh; background: var(--card-bg); border-left: 1px solid var(--border); padding: 1rem; overflow-y: auto; transition: right 0.3s; z-index: 10; }
  .sidebar.open { right: 0; }
  .sidebar h3 { color: var(--accent); margin-bottom: 0.5rem; }
  .sidebar p { font-size: 0.85rem; margin: 0.3rem 0; }
  .filters { position: fixed; top: 10px; left: 10px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; z-index: 10; }
  .filters label { display: block; font-size: 0.8rem; margin: 4px 0; cursor: pointer; }
  .filters input { margin-right: 5px; }
  .topic-selector { position: fixed; top: 10px; left: 50%; transform: translateX(-50%); background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; z-index: 10; }
  .topic-selector select { background: var(--bg); color: var(--fg); border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px; }
  .legend { position: fixed; bottom: 10px; left: 10px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px; font-size: 0.75rem; z-index: 10; }
  .legend-item { display: flex; align-items: center; margin: 3px 0; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; }
</style>
</head>
<body>
<div id="graph-container"><svg id="graph-svg"></svg></div>

<div class="filters" id="filters">
  <strong>Filter by type:</strong>
  <label><input type="checkbox" data-type="fact" checked> Facts</label>
  <label><input type="checkbox" data-type="claim" checked> Claims</label>
  <label><input type="checkbox" data-type="web" checked> Web Evidence</label>
  <label><input type="checkbox" data-type="code" checked> Code Evidence</label>
  <label><input type="checkbox" data-type="ledger" checked> Ledger</label>
</div>

<div class="topic-selector">
  <label>Topic: <select id="topic-select"></select></label>
</div>

<div class="sidebar" id="detail-sidebar">
  <h3 id="detail-title">Select a node</h3>
  <p id="detail-type"></p>
  <p id="detail-status"></p>
  <p id="detail-label" style="white-space:pre-wrap;word-break:break-word;"></p>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#4da6ff"></div>Fact</div>
  <div class="legend-item"><div class="legend-dot" style="background:#00bcd4"></div>Claim</div>
  <div class="legend-item"><div class="legend-dot" style="background:#3cb878"></div>Web</div>
  <div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div>Code</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Ledger</div>
</div>

<script>
const colorMap = { fact: '#4da6ff', claim: '#00bcd4', web: '#3cb878', code: '#a78bfa', ledger: '#f59e0b' };
const edgeStyles = {
  supports: { stroke: '#3cb878', dash: '' },
  conflicts_with: { stroke: '#f87171', dash: '5,5' },
  derived_from: { stroke: '#6b7280', dash: '3,3' },
  supersedes: { stroke: '#f59e0b', dash: '' },
  subsumes: { stroke: '#818cf8', dash: '' },
  same_source: { stroke: '#6b7280', dash: '2,2' },
};

let allNodes = [], allEdges = [], simulation, svg, g;
let currentTransform = d3.zoomIdentity;  // preserve zoom across filter changes
const width = window.innerWidth, height = window.innerHeight;
let currentTopicId = new URLSearchParams(window.location.search).get('topic_id');

async function loadTopics() {
  try {
    const resp = await fetch('/api/topics');
    const data = await resp.json();
    const sel = document.getElementById('topic-select');
    const topics = data.topics || [];
    if (topics.length) {
      sel.innerHTML = topics.map(t => {
        const label = ('#' + t.id + ' ' + (t.summary || '').slice(0, 80))
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
        const selected = String(t.id) === String(currentTopicId || data.current_topic_id || '') ? 'selected' : '';
        return '<option value="' + t.id + '" ' + selected + '>' + label + '</option>';
      }).join('');
      currentTopicId = String(currentTopicId || data.current_topic_id || topics[0].id);
      sel.value = currentTopicId;
      loadGraph(currentTopicId);
    }
  } catch(e) { console.warn('Failed to load topics', e); }
}

function changeTopic(topicId) {
  currentTopicId = String(topicId);
  const params = new URLSearchParams(window.location.search);
  params.set('topic_id', currentTopicId);
  const qs = params.toString();
  window.history.replaceState({}, '', '/graph?' + qs);
  loadGraph(currentTopicId);
}

async function loadGraph(topicId) {
  try {
    const resp = await fetch('/api/graph/' + topicId);
    const data = await resp.json();
    allNodes = data.nodes || [];
    allEdges = data.edges || [];
    renderGraph();
  } catch(e) { console.warn('Failed to load graph', e); }
}

function getVisibleTypes() {
  const types = new Set();
  document.querySelectorAll('#filters input:checked').forEach(cb => types.add(cb.dataset.type));
  return types;
}

function renderGraph() {
  const types = getVisibleTypes();
  const nodes = allNodes.filter(n => types.has(n.type));
  const nodeIds = new Set(nodes.map(n => n.id));
  const edges = allEdges.filter(e => nodeIds.has(e.source?.id || e.source) && nodeIds.has(e.target?.id || e.target));

  d3.select('#graph-svg').selectAll('*').remove();
  svg = d3.select('#graph-svg');
  g = svg.append('g');

  const zoomBehavior = d3.zoom().scaleExtent([0.1, 8]).on('zoom', (event) => {
    currentTransform = event.transform;
    g.attr('transform', event.transform);
  });
  svg.call(zoomBehavior);
  // Restore previous zoom/pan state when re-rendering after filter change
  svg.call(zoomBehavior.transform, currentTransform);
  g.attr('transform', currentTransform);

  const link = g.append('g').selectAll('line').data(edges).enter().append('line')
    .attr('stroke', d => (edgeStyles[d.relation] || edgeStyles.derived_from).stroke)
    .attr('stroke-dasharray', d => (edgeStyles[d.relation] || edgeStyles.derived_from).dash)
    .attr('stroke-width', 1.5)
    .attr('opacity', 0.6);

  link.append('title').text(d => d.relation + (d.confidence != null ? ' (' + d.confidence.toFixed(2) + ')' : ''));

  const node = g.append('g').selectAll('circle').data(nodes).enter().append('circle')
    .attr('r', 8)
    .attr('fill', d => colorMap[d.type] || '#888')
    .attr('stroke', '#fff')
    .attr('stroke-width', 1)
    .attr('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  node.append('title').text(d => d.id + ': ' + d.label);

  node.on('click', (event, d) => {
    const sb = document.getElementById('detail-sidebar');
    sb.classList.add('open');
    document.getElementById('detail-title').textContent = d.id;
    document.getElementById('detail-type').textContent = 'Type: ' + d.type;
    document.getElementById('detail-status').textContent = 'Status: ' + d.status;
    document.getElementById('detail-label').textContent = d.full_label || d.label;
  });

  svg.on('click', (event) => {
    if (event.target.tagName !== 'circle') {
      document.getElementById('detail-sidebar').classList.remove('open');
    }
  });

  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id).distance(100))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide(20));

  simulation.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
  });
}

document.querySelectorAll('#filters input').forEach(cb => cb.addEventListener('change', renderGraph));
document.getElementById('topic-select').addEventListener('change', (e) => changeTopic(e.target.value));
loadTopics();
</script>
</body>
</html>"""


async def handle_graph_page(request):
    return web.Response(text=render_graph_html(), content_type="text/html")


async def handle_graph_data(request):
    topic_id = _int_param(request, "topic_id")
    data = await asyncio.to_thread(build_graph_data, topic_id)
    return web.json_response(data)


# ---------------------------------------------------------------------------
# Phase F.5: Queue endpoints
# ---------------------------------------------------------------------------


async def handle_queue_add(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Body must be a JSON object"}, status=400)
    summary = _str_field(body, "summary")
    detail = _str_field(body, "detail")
    if not summary:
        return web.json_response({"error": "summary is required"}, status=400)
    config = body.get("config")
    try:
        topic_id = await asyncio.to_thread(api.enqueue_topic, summary, detail, config)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"topic_id": topic_id}, status=201)


async def handle_queue_list(request):
    queue = await asyncio.to_thread(api.get_topic_queue)
    return web.json_response(queue)


async def handle_queue_remove(request):
    topic_id = _int_param(request, "topic_id")
    await asyncio.to_thread(api.dequeue_topic, topic_id)
    return web.json_response({"ok": True})


async def handle_queue_reorder(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Body must be a JSON object"}, status=400)
    raw_order = body.get("order", [])
    if not isinstance(raw_order, list):
        return web.json_response({"error": "order must be a list"}, status=400)
    try:
        order = [int(x) for x in raw_order]
    except (ValueError, TypeError):
        return web.json_response(
            {"error": "order must be a list of integers"}, status=400
        )
    await asyncio.to_thread(api.reorder_queue, order)
    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


@web.middleware
async def csrf_middleware(request, handler):
    """Reject cross-origin mutating requests (CSRF protection)."""
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        origin = request.headers.get("Origin", "")
        referer = request.headers.get("Referer", "")
        host = request.headers.get("Host", "")
        port = os.environ.get("ORBIT_WEB_PORT", "8080")

        # Build allowed origins
        allowed_origins = set()
        if host:
            allowed_origins.add(f"http://{host}")
            allowed_origins.add(f"https://{host}")
        allowed_origins.add(f"http://127.0.0.1:{port}")
        allowed_origins.add(f"http://localhost:{port}")

        if origin:
            if origin not in allowed_origins:
                return web.json_response(
                    {"error": "Cross-origin request rejected"}, status=403
                )
        elif referer:
            # Fallback: check Referer when Origin is absent
            if not any(referer.startswith(o) for o in allowed_origins):
                return web.json_response(
                    {"error": "Cross-origin request rejected (referer check)"},
                    status=403,
                )
        else:
            # Neither Origin nor Referer — reject unless it has X-Requested-With
            if not request.headers.get("X-Requested-With"):
                return web.json_response(
                    {"error": "Missing Origin/Referer header"}, status=403
                )
    return await handler(request)


def create_app():
    app = web.Application(middlewares=[csrf_middleware])
    app.router.add_get("/", index)
    app.router.add_get("/api/dashboard", dashboard)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/topics", handle_list_topics)
    # F.1: Topic creation + config
    app.router.add_post("/api/topic/create", handle_create_topic)
    app.router.add_get("/api/topic/{id}/config", handle_get_topic_config)
    app.router.add_put("/api/topic/{id}/config", handle_put_topic_config)
    # F.2: HITL
    app.router.add_post("/api/topic/{id}/resume", handle_resume_topic)
    app.router.add_post("/api/topic/{id}/inject", handle_inject_knowledge)
    app.router.add_get("/api/topic/{id}/pause_status", handle_pause_status)
    app.router.add_get("/api/topic/{id}/mse_review", handle_mse_review)
    app.router.add_get("/api/topic/{id}/mse_report", handle_mse_provenance_report)
    app.router.add_get(
        "/api/topic/{id}/mse_report/markdown", handle_mse_provenance_markdown
    )
    app.router.add_post("/api/topic/{id}/corpus/ingest", handle_ingest_corpus_document)
    app.router.add_post("/api/mse/component/{id}/review", handle_review_mse_component)
    app.router.add_post("/api/mse/diagnostic/{id}/status", handle_update_mse_diagnostic)
    # F.3: Report
    app.router.add_post("/api/topic/{id}/report", handle_trigger_report)
    app.router.add_get("/api/topic/{id}/report", handle_get_report_json)
    app.router.add_get("/api/topic/{id}/report/html", handle_get_report_html)
    app.router.add_get("/api/topic/{id}/report/markdown", handle_get_report_markdown)
    # F.4: Knowledge graph
    app.router.add_get("/graph", handle_graph_page)
    app.router.add_get("/api/graph/{topic_id}", handle_graph_data)
    # F.5: Queue
    app.router.add_post("/api/queue/add", handle_queue_add)
    app.router.add_get("/api/queue", handle_queue_list)
    app.router.add_delete("/api/queue/{topic_id}", handle_queue_remove)
    app.router.add_put("/api/queue/reorder", handle_queue_reorder)
    return app


def main():
    configure_logging()
    host = os.environ.get("ORBIT_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("ORBIT_WEB_PORT", "8080"))
    web.run_app(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
