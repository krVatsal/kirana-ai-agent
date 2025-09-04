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
if 'auto_send' not in state:
    state.auto_send = True
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




with tabs[0]:
    st.header("Customer Voice Assistant")
    st.caption("Speak or type your request (Hindi / English / Hinglish). Gemini powers intent & response when API key is set.")

    # Centered voice UI container
    st.markdown(
        """
        <style>
        .voice-box {border:1px solid #444;padding:1.2rem;border-radius:12px;background:#11111110;}
        .voice-title {font-size:1.05rem;font-weight:600;margin-bottom:0.4rem;}
        .heard {color:#16c60c;font-weight:500;margin-top:0.5rem;}
        </style>
        """,
        unsafe_allow_html=True
    )
    voice_col, controls_col = st.columns([4,1])
    with voice_col:
        with st.container():
            st.markdown('<div class="voice-box"><div class="voice-title">🎙️ Speak</div>', unsafe_allow_html=True)
            st.write("Click mic, allow permission, speak clearly. If nothing appears, check browser permission.")
            new_voice_text = None
            try:
                # dynamic key to allow reset
                comp_key = f'user_stt_{state.stt_component_version}'
                new_voice_text = speech_to_text(language='auto', just_once=True, key=comp_key)
                state.last_stt_error = None
            except Exception as e:
                state.last_stt_error = str(e)
            if new_voice_text:
                st.markdown(f"<div class='heard'>Heard: {new_voice_text}</div>", unsafe_allow_html=True)
            else:
                st.caption("No transcription yet.")
            st.markdown('</div>', unsafe_allow_html=True)
    # Language + Auto-send controls
    lang_label = st.selectbox(
        "Speech Language",
        options=["Hindi (hi-IN)", "English (en-IN)"],
        index=0,
        help="Choose language model for recognition before clicking Start Listening"
    )
    lang_code = "hi-IN" if lang_label.startswith("Hindi") else "en-IN"
    state.auto_send = st.checkbox("Auto-send", value=state.auto_send, help="Automatically process transcript when captured")

    # Install parent listener (once) to capture postMessage from iframe
    components.html("""
    <script>
    if(!window._VOICE_LISTENER_INSTALLED){
      window._VOICE_TRANSCRIPT = window._VOICE_TRANSCRIPT || "";
      window.addEventListener('message', (e)=>{
         if(e.data && e.data.type==='VOICE_TRANSCRIPT' && e.data.value){
            window._VOICE_TRANSCRIPT = e.data.value;
         }
      });
      window._VOICE_LISTENER_INSTALLED = true;
    }
    </script>
    """, height=0)

    # Web Speech API block (runs in iframe, sends transcript via postMessage to parent)
    speech_recognition = f"""
    <script>
    const langCode = '{lang_code}';
    var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = langCode;
    recognition.interimResults = false; // final only
    recognition.maxAlternatives = 1;
    let finalTranscript = '';
    function startRecognition() {{
        const statusEl = document.getElementById('sr_status');
        finalTranscript = '';
        statusEl.innerText = 'Listening...';
        try {{ recognition.start(); }} catch(e) {{ statusEl.innerText = 'Error starting: ' + e.message; return; }}
        recognition.onresult = function(event) {{ finalTranscript = event.results[0][0].transcript; }}
        recognition.onerror = function(e) {{ statusEl.innerText = 'Error: ' + e.error; }}
        recognition.onend = function() {{
            if(finalTranscript) {{
                statusEl.innerText = 'Captured';
                // send to parent
                window.parent.postMessage({{type:'VOICE_TRANSCRIPT', value: finalTranscript}}, '*');
            }} else {{
                statusEl.innerText = 'No speech detected';
            }}
        }}
    }}
    </script>
    <div class='voice-box'>
      <div class='voice-title'>🎙️ Voice Input ({lang_code})</div>
      <button onclick="startRecognition()">Start Listening</button>
      <span id='sr_status' style='margin-left:8px;font-size:0.8rem;opacity:0.75;'>Idle</span>
      <div style='margin-top:0.7rem;font-size:0.75rem;opacity:0.65;'>Auto-send after you stop speaking is { 'ON' if state.auto_send else 'OFF' }.</div>
    </div>
    """
    components.html(speech_recognition, height=150)
    recognized_text = streamlit_js_eval(
        js_expressions="window._VOICE_TRANSCRIPT",
        key="speech_eval"
    )

    # Process recognized speech
    auto_fired = False
    if state.auto_send and recognized_text and recognized_text.strip() and recognized_text != state.last_voice_processed:
        with st.spinner("Processing voice..."):
            state.chat.append({"role":"user","text":recognized_text})
            save_chat('user', recognized_text)
            result = process_user_message(recognized_text)
            reply = result.get('response_text', '(No response)')
            state.chat.append({"role":"assistant","text":reply})
            save_chat('assistant', reply, result.get('order_id'))
            speak(reply)
            state.last_voice_processed = recognized_text
            auto_fired = True

    # Manual send button if auto-send disabled
    if (not state.auto_send) and recognized_text and recognized_text.strip() and recognized_text != state.last_voice_processed:
        if st.button("Send Recognized Text", type="primary"):
            with st.spinner("Processing voice..."):
                state.chat.append({"role":"user","text":recognized_text})
                save_chat('user', recognized_text)
                result = process_user_message(recognized_text)
                reply = result.get('response_text', '(No response)')
                state.chat.append({"role":"assistant","text":reply})
                save_chat('assistant', reply, result.get('order_id'))
                speak(reply)
                state.last_voice_processed = recognized_text

    # Resend last
    resend_col, clear_col = st.columns([1,1])
    with resend_col:
        if st.button("Resend Last", disabled=not state.last_voice_processed):
            lv = state.last_voice_processed
            if lv:
                state.chat.append({"role":"user","text":lv})
                save_chat('user', lv)
                result = process_user_message(lv)
                reply = result.get('response_text', '(No response)')
                state.chat.append({"role":"assistant","text":reply})
                save_chat('assistant', reply, result.get('order_id'))
                speak(reply)
    with clear_col:
        if st.button("Clear Transcript"):
            state.last_voice_processed = None
            st.experimental_rerun() if hasattr(st,'experimental_rerun') else st.rerun()

    manual_text = st.text_input("Or type instead:")
    if st.button("Send Text", use_container_width=True, disabled=not manual_text.strip()):
        with st.spinner("Processing text..."):
            state.chat.append({"role":"user","text":manual_text})
            save_chat('user', manual_text)
            result = process_user_message(manual_text)
            reply = result.get('response_text', '(No response)')
            state.chat.append({"role":"assistant","text":reply})
            save_chat('assistant', reply, result.get('order_id'))
            st.markdown(f"**AI:** {reply}")
            speak(reply)

    with st.expander("Diagnostics", expanded=False):
        st.json({
            "recognized_text": recognized_text if recognized_text else None,
            "auto_send": state.auto_send,
            "last_voice_processed": state.last_voice_processed,
            "language_code": lang_code,
            "model_loaded": bool(model),
            "orders_count": len(state.orders),
            "auto_fired": auto_fired,
        })
        if getattr(state, 'last_raw_model_output', None):
            st.text_area("Raw Model Output", state.last_raw_model_output, height=140)

# ---------------- (Appended) Conversation + Shopkeeper Dashboard -----------------
st.divider()
st.subheader("Conversation")
for _msg in reversed(state.chat[-25:]):
    if _msg['role'] == 'user':
        st.markdown(f"🧑 **You:** {_msg['text']}")
    else:
        st.markdown(f"🤖 **Agent:** {_msg['text']}")

with tabs[1]:
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
            if st.button("Progress Status Simulation"):
                update_statuses()
                force_rerun()
        with col_b:
            if st.button("Reload From DB"):
                state.orders = load_orders()
                state.chat = load_chat()
                recompute_inventory_from_orders()
                force_rerun()
        with col_c:
            if st.button("Recompute Inventory"):
                recompute_inventory_from_orders()
                st.success("Inventory recomputed from all orders.")
    else:
        st.info("No orders yet.")

    if not model:
        st.warning("Gemini API key not configured. Set GOOGLE_API_KEY for full AI responses.")

st.caption("Prototype: Voice via Web Speech API; AI reasoning Gemini; TTS gTTS; persistence SQLite (data.db).")
