import os
import json
import re
import io
from datetime import datetime
import streamlit as st
from langgraph.graph import StateGraph, END
# Removed mic_recorder speech_to_text (replaced with Web Speech API JS)
from gtts import gTTS
import streamlit.components.v1 as components
from dotenv import load_dotenv 
from streamlit_js_eval import streamlit_js_eval
load_dotenv()

import google.generativeai as genai

from storage import init_db, save_order, update_order_status, load_orders, save_chat, load_chat, price_for_item


# ---------------- Inventory (only hardcoded domain data) -----------------
INVENTORY = {
    'milk': {'hindi': ['doodh'], 'qty': 10, 'unit': 'packet'},
    'bread': {'hindi': ['bread'], 'qty': 5, 'unit': 'loaf'},
    'rice': {'hindi': ['chawal'], 'qty': 8, 'unit': 'kilo'},
    'maggi': {'hindi': ['maggi'], 'qty': 12, 'unit': 'packet'},
}

# ---------------- Session State Initialization -----------------
state = st.session_state
if 'orders' not in state:
    state.orders = []
if 'inventory' not in state:
    state.inventory = {k: v['qty'] for k, v in INVENTORY.items()}
if 'chat' not in state:
    state.chat = []  # list of {role:'user'|'assistant', 'text': str}
if 'order_counter' not in state:
    state.order_counter = 1
if 'chat_loaded' not in state:
    init_db()
    # load persisted
    loaded_orders = load_orders()
    if loaded_orders:
        state.orders = loaded_orders
        if loaded_orders[-1]['id'] >= state.order_counter:
            state.order_counter = loaded_orders[-1]['id'] + 1
    state.chat = load_chat()
    state.chat_loaded = True
    # After loading historical orders, recompute inventory so it reflects past orders
    base_inventory = {k: v['qty'] for k, v in INVENTORY.items()}
    for _ord in state.orders:
        for item_name, qty in _ord['items']:
            if item_name in base_inventory:
                base_inventory[item_name] = max(0, base_inventory[item_name] - qty)
    state.inventory = base_inventory
if 'last_voice_processed' not in state:
    state.last_voice_processed = None
if 'last_voice_session_id' not in state:
    state.last_voice_session_id = None
if 'last_voice_text' not in state:
    state.last_voice_text = None
if 'last_stt_error' not in state:
    state.last_stt_error = None
if 'stt_component_version' not in state:
    state.stt_component_version = 0
# ---------------- Gemini Setup -----------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY and genai:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        model = None
else:
    model = None

# ---------------- Utility: Text-To-Speech -----------------


def speak(text: str):
    try:
        lang = 'hi' if re.search(r'[\u0900-\u097F]', text) else 'en'
        bio = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(bio)
        bio.seek(0)
        st.audio(bio.read(), format='audio/mp3')
    except Exception as e:
        st.warning(f"TTS failed: {e}")

# ---------------- Gemini Parsing -----------------
PROMPT_TEMPLATE = """
You are an AI assistant for a small Indian kirana (grocery) store. Understand multilingual (Hinglish, Hindi, English) user utterances.
Task: Given the USER_MESSAGE and current INVENTORY and ORDERS, output a concise JSON ONLY (no extra text) with keys:
intent: one of [order, inventory_check, status, greeting, unknown]
items: list of objects {{name, qty}} only if intent=order (normalize names to: {valid_items})
response_text: A natural reply in the SAME language/style as user (mix if user mixes). For order: confirm availability, price estimate (~just sum qty * 10 for demo), and delivery ETA 30 minutes. If insufficient stock, propose available qty.
If status intent: summarize latest undelivered order progress realistically.
If inventory_check: answer availability.
If greeting: greet and offer help.
If unknown: ask for clarification.
Constraints: Return ONLY JSON. Do NOT hallucinate items not in inventory.

INVENTORY (name: remaining_qty with unit):
{inventory_block}

ACTIVE ORDERS:
{orders_block}

USER_MESSAGE: "{user_message}"
"""

def gemini_parse(user_text: str):
    inventory_block = '\n'.join([
        f"{name}: {state.inventory.get(name,0)} {INVENTORY[name]['unit']} (orig {INVENTORY[name]['qty']})" for name in INVENTORY
    ])
    orders_block = 'None' if not state.orders else '\n'.join([
        f"Order#{o['id']} status={o['status']} items={o['items']}" for o in state.orders
    ])
    prompt = PROMPT_TEMPLATE.format(
        valid_items=', '.join(INVENTORY.keys()),
        inventory_block=inventory_block,
        orders_block=orders_block,
        user_message=user_text
    )
    def extract_json_block(raw: str):
        # Remove code fences
        raw = raw.strip()
        if raw.startswith('```'):
            raw = re.sub(r'^```[a-zA-Z0-9]*', '', raw).strip()
        if raw.endswith('```'):
            raw = raw[:-3].strip()
        # Find first '{' and attempt to balance braces
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1 or end < start:
            return None
        candidate = raw[start:end+1]
        # Simple brace balance check
        stack = 0
        for ch in candidate:
            if ch == '{':
                stack += 1
            elif ch == '}':
                stack -= 1
                if stack < 0:
                    return None
        if stack != 0:
            return None
        return candidate

    last_error = None
    for attempt in range(2):  # first try original prompt, second forced JSON if needed
        try:
            resp = model.generate_content(prompt if attempt == 0 else prompt + "\nReturn ONLY raw minified JSON starting with '{' and nothing else.")
            raw_text = resp.text or ''
            cleaned = extract_json_block(raw_text) or raw_text
            data = json.loads(cleaned)
            # Optionally store raw for debug
            state.last_raw_model_output = raw_text
            return data
        except Exception as e:
            last_error = e
            state.last_raw_model_output = locals().get('raw_text', '')
            continue
    return {"intent": "unknown", "items": [], "response_text": f"Parsing error: {last_error}"}

# ---------------- Order Handling -----------------
def apply_order(items, raw_request:str, response_text:str):
    unavailable = []
    applied_pairs = []
    detailed_items = []
    total_amount = 0.0
    for it in items:
        name = it.get('name')
        qty = int(it.get('qty',1) or 1)
        if name not in state.inventory:
            unavailable.append({"name": name, "reason": "not_found"})
            continue
        if state.inventory[name] < qty:
            unavailable.append({"name": name, "reason": f"only {state.inventory[name]} left"})
        else:
            state.inventory[name] -= qty
            applied_pairs.append((name, qty))
            unit_price = price_for_item(name)
            line_total = unit_price * qty
            total_amount += line_total
            detailed_items.append({"name": name, "qty": qty, "unit_price": unit_price, "line_total": line_total})
    order_id = None
    if applied_pairs:
        order_id = state.order_counter
        state.order_counter += 1
        state.orders.append({"id": order_id, "items": applied_pairs, "status": "processing", "total_amount": total_amount})
        save_order(order_id, 'processing', detailed_items, raw_request, response_text, total_amount)
    return applied_pairs, unavailable, order_id

def update_statuses():
    for o in state.orders:
        before = o['status']
        if before == 'processing':
            o['status'] = 'out-for-delivery'
        elif before == 'out-for-delivery':
            o['status'] = 'delivered'
        if o['status'] != before:
            update_order_status(o['id'], o['status'])

def recompute_inventory_from_orders():
    """Rebuild current inventory from the original INVENTORY minus all applied order items."""
    rebuilt = {k: v['qty'] for k, v in INVENTORY.items()}
    for ord_row in state.orders:
        for item_name, qty in ord_row.get('items', []):
            if item_name in rebuilt:
                rebuilt[item_name] = max(0, rebuilt[item_name] - qty)
    state.inventory = rebuilt

# -------- Rerun helper (handles Streamlit version differences) --------
def force_rerun():
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()

def _build_graph():


    # State will be a dict: { user_text, parsed }
    graph = StateGraph(dict)

    def gemini_node(state_dict: dict):
        user_text = state_dict.get('user_text', '')
        parsed = gemini_parse(user_text)
        # attach original user text for downstream nodes
        parsed['__user_text'] = user_text
        state_dict['parsed'] = parsed
        return state_dict

    def order_node(state_dict: dict):
        parsed = state_dict.get('parsed',{})
        if parsed.get('intent') == 'order':
            user_text_local = parsed.get('__user_text','')
            applied, unavailable, oid = apply_order(parsed.get('items', []), user_text_local, parsed.get('response_text',''))
            parsed['applied_items'] = applied
            parsed['unavailable'] = unavailable
            if oid:
                parsed['order_id'] = oid
            # Optional refinement when unavailable
            if unavailable:
                unavailable_desc = ', '.join([f"{u['name']} ({u['reason']})" for u in unavailable])
                clarification_prompt = f"User tried ordering items with issues: {unavailable_desc}. Create a concise apology + suggestion in same language."  # noqa
                if model:
                    try:
                        alt = model.generate_content(clarification_prompt).text.strip()
                        parsed['response_text'] += "\n" + alt
                    except Exception:
                        pass
        state_dict['parsed'] = parsed
        return state_dict

    def route_after_gemini(state_dict: dict):
        intent = state_dict.get('parsed',{}).get('intent')
        if intent == 'order':
            return 'order'
        return END

    graph.add_node('gemini', gemini_node)
    graph.add_node('order', order_node)
    graph.set_entry_point('gemini')
    graph.add_conditional_edges('gemini', route_after_gemini, {'order': 'order', END: END})
    graph.add_edge('order', END)
    return graph.compile()

APP_GRAPH = _build_graph()

def process_user_message(user_text: str):
    if APP_GRAPH is None:
        # Fallback sequential processing if langgraph not available
        parsed = gemini_parse(user_text)
        parsed['__user_text'] = user_text
        if parsed.get('intent') == 'order':
            applied, unavailable, oid = apply_order(parsed.get('items', []), user_text, parsed.get('response_text',''))
            parsed['applied_items'] = applied
            parsed['unavailable'] = unavailable
            if oid:
                parsed['order_id'] = oid
        return parsed
    final_state = APP_GRAPH.invoke({'user_text': user_text})
    return final_state.get('parsed', {})




# ---------------- UI Layout -----------------
st.set_page_config(page_title="Kirana AI Agent (Voice AI)", layout="wide")
tabs = st.tabs(["Customer App", "Shopkeeper Dashboard"])  # Could add Analytics later

def render_shopkeeper_dashboard(key_prefix: str = "shop_tab"):
    st.header("Shopkeeper Dashboard")
    st.subheader("Inventory")
    st.table([
        { 'Item': k, 'Remaining Qty': state.inventory.get(k,0), 'Unit': INVENTORY[k]['unit'] }
        for k in INVENTORY
    ])

    st.subheader("Orders")
    if state.orders:
        for _o in state.orders[-20:][::-1]:
            _total = _o.get('total_amount', 0.0)
            st.markdown(
                f"**Order #{_o['id']}** - Status: {_o['status']} | Items: " +
                ', '.join([f"{qty} {name}" for name, qty in _o['items']]) +
                f" | Total: ₹{_total:.2f}"
            )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Progress Status Simulation", key=f"{key_prefix}_prog_status"):
                update_statuses()
                force_rerun()
        with col_b:
            if st.button("Reload From DB", key=f"{key_prefix}_reload_db"):
                state.orders = load_orders()
                state.chat = load_chat()
                recompute_inventory_from_orders()
                force_rerun()
        with col_c:
            if st.button("Recompute Inventory", key=f"{key_prefix}_recompute_inv"):
                recompute_inventory_from_orders()
                st.success("Inventory recomputed from all orders.")
    else:
        st.info("No orders yet.")

    if not model:
        st.warning("Gemini API key not configured. Set GOOGLE_API_KEY for full AI responses.")




with tabs[0]:
    st.header("Customer Assistant")
    st.caption("Talk or type in Hindi / English / Hinglish. Orders & queries handled by the AI.")

    # Global CSS for cleaner UI
    st.markdown(
        """
        <style>
    /* Layout padding reduction */
    .block-container {padding-top:0.6rem !important;}
    .better-button {position:relative;display:inline-flex;align-items:center;gap:.55rem;background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff !important;border:1px solid #1e40af;padding:0.60rem 1.05rem;border-radius:10px;font-weight:600;font-size:.85rem;letter-spacing:.3px;cursor:pointer;box-shadow:0 2px 4px rgba(0,0,0,.4),0 0 0 2px rgba(37,99,235,.18);transition:background .18s,border-color .18s,transform .18s,box-shadow .25s;}
    .better-button:hover:not([disabled]) {background:linear-gradient(135deg,#1d4ed8,#2563eb);transform:translateY(-1px);}
    .better-button:active:not([disabled]) {transform:translateY(0);background:#1e3a8a;}
    .better-button:focus-visible {outline:2px solid #93c5fd;outline-offset:3px;}
    .better-button[disabled] {opacity:.55;cursor:not-allowed;}
    .better-button[data-state='listening'] {background:#dc2626;border-color:#b91c1c;box-shadow:0 0 0 2px rgba(239,68,68,.35),0 4px 10px -2px rgba(239,68,68,.4);}
    .better-button[data-state='listening'] .btn-label:before {content:'\25CF';display:inline-block;color:#fecaca;animation:blink 1s linear infinite;margin-right:4px;font-size:.7rem;}
    .better-button[data-state='listening']::after {content:"";position:absolute;inset:-6px;border:2px solid rgba(239,68,68,.45);border-radius:14px;animation:pulse 1.2s ease-in-out infinite;}
    @keyframes pulse {0% {transform:scale(.92);opacity:1;}70% {transform:scale(1.15);opacity:0;}100% {opacity:0;}}
    @keyframes blink {0%,60% {opacity:1;}61%,100% {opacity:.2;}}
    .better-button .btn-label {display:inline-block;}
    /* Voice input container */
    .voice-box {border:1px solid #30363d;padding:0.9rem 1.05rem;border-radius:12px;background:#1d232a;color:#d2d8df;}
    .voice-title {font-size:1.0rem;font-weight:600;margin-bottom:0.35rem;display:flex;align-items:center;gap:.4rem;color:#f1f5f9;}
    .voice-box button {background:#2563eb;color:#ffffff !important;border:1px solid #1d4ed8;padding:0.55rem 0.95rem;border-radius:8px;font-weight:600;cursor:pointer;transition:background .15s,border-color .15s;}
    .voice-box button:hover {background:#1d4ed8;border-color:#1e40af;}
    .voice-box button:active {background:#1e3a8a;border-color:#1e3a8a;}
    #sr_status {color:#94a3b8;}
        .chat-wrap {border:1px solid #30363d;border-radius:12px;padding:.75rem;background:#0f1115;max-height:480px;overflow-y:auto;}
        .msg-user {background:#2563eb10;border:1px solid #2563eb30;padding:.55rem .75rem;border-radius:10px;margin-bottom:.6rem;}
        .msg-ai {background:#10b98110;border:1px solid #10b98130;padding:.55rem .75rem;border-radius:10px;margin-bottom:.6rem;}
        .badge {padding:2px 8px;border-radius:12px;font-size:.7rem;font-weight:600;display:inline-block;}
        .status-processing {background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b55;}
        .status-out-for-delivery {background:#6366f122;color:#6366f1;border:1px solid #6366f155;}
        .status-delivered {background:#10b98122;color:#10b981;border:1px solid #10b98155;}
        ::-webkit-scrollbar {width:8px;}::-webkit-scrollbar-thumb {background:#333;border-radius:6px;}
        </style>
        """,
        unsafe_allow_html=True
    )

    lang_label = st.selectbox(
        "Language",
        options=["Hindi (hi-IN)", "English (en-IN)"],
        index=0,
        help="Choose before recording"
    )
    lang_code = "hi-IN" if lang_label.startswith("Hindi") else "en-IN"

    # Install parent listener (once) to capture postMessage from iframe
    components.html("""
    <script>
    if(!window._VOICE_LISTENER_INSTALLED){
    window._VOICE_TRANSCRIPT_JSON = window._VOICE_TRANSCRIPT_JSON || '';
    window.addEventListener('message', (e)=>{
       if(e.data && e.data.type==='VOICE_TRANSCRIPT' && e.data.value){
        try{ window._VOICE_TRANSCRIPT_JSON = JSON.stringify(e.data.value);}catch(err){console.warn(err);} }
    });
    window._VOICE_LISTENER_INSTALLED = true;
    }
    </script>
    """, height=10)

    # Web Speech API block (runs in iframe, sends transcript via postMessage to parent)
    speech_recognition = """
    <script>
    const langCode = '%s';
    let recognition; let listening=false;
    function init(){
        if(!recognition){
            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = langCode;
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            recognition.onresult = (e)=>{ window._LAST_FINAL = e.results[0][0].transcript; };
            recognition.onerror = (e)=>{ setStatus('Error: '+ e.error); listening=false; };
            recognition.onend = ()=>{ if(listening){ listening=false; finalize(); } };
        }
    }
    function setStatus(s){ const el=document.getElementById('sr_status'); if(el) el.innerText=s; }
    function start(){
        init();
        if(listening) return;
        window._LAST_FINAL='';
        listening=true;
        setStatus('Listening...');
        try { recognition.start(); } catch(err){ setStatus('Start err: '+ err.message); listening=false; }
    }
    function finalize(){
        const t = window._LAST_FINAL || '';
        const c = document.getElementById('captured_text');
        if(t){
            if(c) c.textContent = t;
            setStatus('Captured');
            const payload = { id: Date.now().toString() + '-' + Math.random().toString(36).slice(2,8), text: t };
            window.parent.postMessage({ type:'VOICE_TRANSCRIPT', value: payload }, '*');
        } else {
            setStatus('No speech');
            if(c) c.textContent='';
        }
    }
    </script>
    <div class='voice-box'>
      <button class="better-button" onclick="start()"><span class='btn-label'>Start Listening</span></button>
      <span id='sr_status' style='margin-left:8px;font-size:0.8rem;opacity:0.75; color:#ffffff'>Idle</span>
      <div id='captured_text' style='margin-top:6px;font-size:0.75rem;color:#cbd5e1;min-height:16px;word-wrap:break-word;'></div>
      <div style='margin-top:0.6rem;font-size:0.65rem;opacity:0.55;color:#ffffff'>Auto-send enabled.</div>
    </div>
    """ % lang_code
    components.html(speech_recognition, height=140)
    recognized_payload = streamlit_js_eval(
        js_expressions="window._VOICE_TRANSCRIPT_JSON",
        key="speech_eval"
    )

    # Process recognized speech (final) automatically
# Process recognized speech (final) automatically
    # Auto-send using unique session id payload
    if recognized_payload:
        try:
            data_obj = json.loads(recognized_payload)
            sid = data_obj.get('id')
            txt = (data_obj.get('text') or '').strip()
        except Exception:
            sid = None; txt = ''
        if sid and txt and sid != state.last_voice_session_id:
            with st.spinner("Processing voice..."):
                state.chat.append({"role":"user","text":txt})
                save_chat('user', txt)
                parsed = process_user_message(txt)
                reply = parsed.get('response_text','(No response)')
                state.chat.append({"role":"assistant","text":reply})
                save_chat('assistant', reply, parsed.get('order_id'))
                speak(reply)
                state.last_voice_session_id = sid
                state.last_voice_text = txt


    # Removed manual send / resend / clear buttons for simplicity

    manual_col, send_col = st.columns([4,1])
    with manual_col:
        manual_text = st.text_input("Type a message", placeholder="e.g. 2 doodh 1 bread status?")
    with send_col:
        if st.button("Send", disabled=not manual_text.strip(), use_container_width=True):
            with st.spinner("Thinking..."):
                state.chat.append({"role":"user","text":manual_text})
                save_chat('user', manual_text)
                parsed = process_user_message(manual_text)
                reply = parsed.get('response_text', '(No response)')
                state.chat.append({"role":"assistant","text":reply})
                save_chat('assistant', reply, parsed.get('order_id'))
                speak(reply)

    # Conversation (now only inside Customer App tab)
    st.divider()
    st.subheader("Conversation")
    chat_container = st.container()
    with chat_container:
        for _msg in reversed(state.chat[-40:]):
            if _msg['role'] == 'user':
                st.markdown(f"<div class='msg-user'><strong>You:</strong> {_msg['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='msg-ai'><strong>AI:</strong> {_msg['text']}</div>", unsafe_allow_html=True)

    # Fallback inline Shopkeeper dashboard (optional quick view)
    with st.expander("Shopkeeper Dashboard (Quick View)", expanded=False):
        render_shopkeeper_dashboard("inline")

# ---------------- Shopkeeper Dashboard -----------------
with tabs[1]:
    render_shopkeeper_dashboard("tab")

st.caption("Prototype: Voice via Web Speech API; AI reasoning Gemini; TTS gTTS; persistence SQLite (data.db).")
