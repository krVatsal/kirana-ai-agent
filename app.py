import os
import json
import re
import io
from datetime import datetime
import streamlit as st
from langgraph.graph import StateGraph, END
from gtts import gTTS
import streamlit.components.v1 as components
from dotenv import load_dotenv
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

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
if 'manual_text_input' not in state:
    state.manual_text_input = ''
if 'msg_input_prefill' not in state:
    state.msg_input_prefill = ''
if 'clear_msg_input' not in state:
    state.clear_msg_input = False
    state.stt_component_version = 0
# ---------------- Gemini Setup (support st.secrets) -----------------
def _fetch_api_key():
    # Priority: st.secrets (flat), st.secrets["google"]["api_key"], then environment
    try:
        if 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets['GOOGLE_API_KEY']
    except Exception:
        pass
    return os.getenv('GOOGLE_API_KEY')

API_KEY = _fetch_api_key()
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

# ---------------- Low Stock Monitoring Agent -----------------
def check_low_stock_and_alert():
    """Auto stock monitoring agent - alerts when inventory is low"""
    low_stock_items = []
    for item, stock in state.inventory.items():
        # Consider stock low if less than 5 units
        if stock < 5:
            low_stock_items.append(f"{item} ({stock} left)")
    
    if low_stock_items:
        alert_msg = f"⚠️ LOW STOCK ALERT: {', '.join(low_stock_items)}. Please restock these items."
        
        # Add alert to chat if not already present in last 3 messages
        if not any(msg.get('text', '').startswith('⚠️ LOW STOCK ALERT') for msg in state.chat[-3:]):
            state.chat.append({"role": "assistant", "text": alert_msg})
            save_chat('assistant', alert_msg)
            return True
    return False

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
        
        # Auto check for low stock after order
        check_low_stock_and_alert()
    
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
    
    # Auto check for low stock after inventory rebuild
    check_low_stock_and_alert()

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
st.set_page_config(page_title="Kirana AI Agent", layout="wide")
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
    # WhatsApp-like Chat Interface
    st.markdown(
        """
        <style>
        /* WhatsApp-style Chat UI */
        .main-container {
            max-width: 800px;
            margin: 0 auto;
            background: #1a1a1a;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .chat-header {
            background: #128c7e;
            color: white;
            padding: 15px 20px;
            font-weight: bold;
            font-size: 18px;
        }
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 10px;
            background: #0a0a0a;
            background-image: url("data:image/svg+xml,%3csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3e%3cg fill='none' fill-rule='evenodd'%3e%3cg fill='%23333333' fill-opacity='0.05'%3e%3ccircle cx='30' cy='30' r='4'/%3e%3c/g%3e%3c/g%3e%3c/svg%3e");
        }
        .msg-bubble {
            display: inline-block;
            min-width: 80px;
            max-width: min(70%, 400px);
            margin: 5px 0;
            padding: 8px 12px;
            border-radius: 18px;
            word-wrap: break-word;
            position: relative;
            width: fit-content;
        }
        .msg-user {
            background: #005c4b;
            color: white;
            text-align: left;
        }
        .msg-ai {
            background: #1f2937;
            color: white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        .input-container {
            display: flex;
            align-items: center;
            padding: 10px;
            background: #2a2a2a;
            gap: 8px;
        }
        .voice-btn {
            background: #25d366 !important;
            border: none !important;
            border-radius: 50% !important;
            width: 45px !important;
            height: 45px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            color: white !important;
            font-size: 18px !important;
            cursor: pointer !important;
            transition: all 0.2s !important;
            box-shadow: 0 2px 8px rgba(37, 211, 102, 0.3) !important;
        }
        .voice-btn:hover {
            background: #128c7e !important;
            transform: scale(1.05) !important;
            box-shadow: 0 4px 12px rgba(18, 140, 126, 0.4) !important;
        }
        .voice-btn.listening {
            background: #ff4444 !important;
            animation: pulse 1.5s infinite !important;
        }
        .send-btn {
            background: #25d366 !important;
            border: none !important;
            border-radius: 50% !important;
            width: 45px !important;
            height: 45px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            color: white !important;
            font-size: 18px !important;
            cursor: pointer !important;
            transition: all 0.2s !important;
            box-shadow: 0 2px 8px rgba(37, 211, 102, 0.3) !important;
        }
        .send-btn:hover {
            background: #128c7e !important;
            transform: scale(1.05) !important;
            box-shadow: 0 4px 12px rgba(18, 140, 126, 0.4) !important;
        }
        .send-btn:disabled {
            background: #ccc !important;
            cursor: not-allowed !important;
            box-shadow: none !important;
        }
        /* Override Streamlit button styles */
        .stButton > button {
            background: #25d366 !important;
            color: white !important;
            border: none !important;
            border-radius: 50% !important;
            width: 45px !important;
            height: 45px !important;
            font-size: 18px !important;
            transition: all 0.3s ease !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 2px 8px rgba(37, 211, 102, 0.3) !important;
            min-height: 45px !important;
        }
        .stButton > button:hover {
            background: #128c7e !important;
            transform: scale(1.05) !important;
            box-shadow: 0 4px 12px rgba(18, 140, 126, 0.4) !important;
        }
        .stButton > button:disabled {
            background: #ccc !important;
            color: #999 !important;
            box-shadow: none !important;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .stTextInput > div > div > input {
            border-radius: 25px !important;
            border: 1px solid #ddd !important;
            padding: 12px 16px !important;
            font-size: 14px !important;
        }
        .voice-status {
            font-size: 12px;
            color: #666;
            margin-left: 10px;
        }
        /* Hide default streamlit elements */
        .stSelectbox > label, .stTextInput > label {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # WhatsApp-like container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">🛒 Kirana AI Assistant</div>', unsafe_allow_html=True)
    
    # Language selector (compact)
    lang_label = st.selectbox("", options=["Hindi (hi-IN)", "English (en-IN)"], index=0, label_visibility="collapsed")
    lang_code = "hi-IN" if lang_label.startswith("Hindi") else "en-IN"

    # Clear request from previous send
    if state.clear_msg_input:
        if 'msg_input_widget' in st.session_state:
            del st.session_state['msg_input_widget']
        state.clear_msg_input = False
        if state.msg_input_prefill:
            state.msg_input_prefill = ''

    # Determine initial value only if widget not yet created
    if 'msg_input_widget' not in st.session_state:
        initial_value = state.msg_input_prefill or ''
    else:
        initial_value = st.session_state.get('msg_input_widget', '')

    # Chat messages container
    for _msg in state.chat[-30:]:  # Show last 30 messages
        if _msg['role'] == 'user':
            st.markdown(f'<div style="display: flex; justify-content: flex-end; margin: 5px 0;"><div class="msg-bubble msg-user">{_msg["text"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="display: flex; justify-content: flex-start; margin: 5px 0;"><div class="msg-bubble msg-ai">{_msg["text"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # WhatsApp-style input bar
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 8, 1])
    
    # Voice button using Bokeh - working solution!
    with col1:
        stt_button = Button(label="🎤", width=45, height=45)
        
        stt_button.js_on_event("button_click", CustomJS(code=f"""
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = '{lang_code}';
         
            recognition.onresult = function (e) {{
                var value = "";
                for (var i = e.resultIndex; i < e.results.length; ++i) {{
                    if (e.results[i].isFinal) {{
                        value += e.results[i][0].transcript;
                    }}
                }}
                if ( value != "") {{
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {{detail: value}}));
                }}
            }}
            recognition.start();
            """))

        voice_result = streamlit_bokeh_events(
            stt_button,
            events="GET_TEXT",
            key="listen",
            refresh_on_update=False,
            override_height=50,
            debounce_time=0)

        # Handle voice result and auto-send
        if voice_result:
            if "GET_TEXT" in voice_result:
                voice_text = voice_result.get("GET_TEXT").strip()
                if voice_text:
                    st.success(f"🎤 Heard: {voice_text}")
                    # Auto-send the voice input
                    with st.spinner("Processing voice input..."):
                        state.chat.append({"role":"user","text":voice_text})
                        save_chat('user', voice_text)
                        parsed = process_user_message(voice_text)
                        reply = parsed.get('response_text', '(No response)')
                        state.chat.append({"role":"assistant","text":reply})
                        save_chat('assistant', reply, parsed.get('order_id'))
                        speak(reply)
                    force_rerun()
    
    # Text input
    with col2:
        manual_text = st.text_input(
            "",
            placeholder="Type a message...",
            value=initial_value,
            key='msg_input_widget',
            label_visibility="collapsed"
        )
        state.msg_input_prefill = ''  # reset prefill so next rerun doesn't overwrite user edits

    # Send button  
    with col3:
        send_clicked = st.button("send", key="send_btn", disabled=not manual_text.strip(), help="Send message")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close main container
    
    # Process send
    if send_clicked and manual_text.strip():
        user_msg = manual_text.strip()
        with st.spinner("Thinking..."):
            state.chat.append({"role":"user","text":user_msg})
            save_chat('user', user_msg)
            parsed = process_user_message(user_msg)
            reply = parsed.get('response_text', '(No response)')
            state.chat.append({"role":"assistant","text":reply})
            save_chat('assistant', reply, parsed.get('order_id'))
            speak(reply)
        state.clear_msg_input = True
        force_rerun()




# ---------------- Shopkeeper Dashboard -----------------
with tabs[1]:
    render_shopkeeper_dashboard("tab")

st.caption("Prototype: Voice via Web Speech API; AI reasoning Gemini; TTS gTTS; persistence SQLite (data.db).")
