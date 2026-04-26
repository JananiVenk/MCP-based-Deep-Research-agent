import streamlit as st
from src.agent import run_agent, get_cached_answer

st.set_page_config(
    page_title="MCP Research Agent",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&family=Source+Sans+3:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #FAF8F3;
  }

  .main { background-color: #FAF8F3; }
  .block-container { max-width: 760px; padding-top: 2rem; }

  /* Header */
  .agent-header {
    background: #1B3A6B;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .agent-header-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: #FFFFFF;
    margin: 0;
  }
  .agent-header-caption {
    font-size: 11px;
    color: #A8BDD8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 4px;
  }

  /* Chat messages */
  .stChatMessage {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 10px;
    margin-bottom: 10px;
  }

  [data-testid="stChatMessageContent"] p {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 15px;
    color: #1C1917;
    line-height: 1.7;
  }

  /* Cache badge */
  .cache-badge {
    display: inline-block;
    background: #EFF3FA;
    color: #1B3A6B;
    border: 1px solid #C5D4EC;
    border-radius: 20px;
    font-size: 11px;
    padding: 2px 10px;
    margin-bottom: 8px;
  }

  /* Source tags */
  .source-tag {
    display: inline-block;
    background: #EFF3FA;
    color: #1B3A6B;
    border: 1px solid #C5D4EC;
    border-radius: 20px;
    font-size: 11px;
    padding: 2px 10px;
    margin-right: 5px;
    margin-top: 6px;
  }

  /* Chat input */
  .stChatInputContainer {
    border-top: 1px solid #E8E4DC;
    padding-top: 12px;
    background: #FAF8F3;
  }
  .stChatInputContainer textarea {
    font-family: 'Playfair Display', serif !important;
    background: #FFFFFF !important;
    border: 1px solid #E8E4DC !important;
    border-radius: 8px !important;
    color: #1C1917 !important;
    font-size: 15px !important;
  }
  .stChatInputContainer button {
    background: #1B3A6B !important;
    border-radius: 8px !important;
  }

  /* Spinner */
  .stSpinner > div {
    border-top-color: #1B3A6B !important;
  }

  /* Status indicator */
  .status-bar {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
    font-size: 11px;
    color: #6B6357;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #1B3A6B;
    margin-right: 5px;
    vertical-align: middle;
  }

  hr { border: none; border-top: 1px solid #E8E4DC; margin: 16px 0; }
  #MainMenu, footer, header { visibility: hidden; }
</style>

<div class="agent-header">
  <div>
    <div class="agent-header-title">MCP Research Agent</div>
    <div class="agent-header-caption">NewsAPI &nbsp;·&nbsp; arXiv &nbsp;·&nbsp; DuckDuckGo &nbsp;·&nbsp; Gemini</div>
  </div>
  <span style="font-size: 28px;">🔍</span>
</div>

<div class="status-bar">
  <span><span class="status-dot"></span>3 MCP servers active</span>
  <span><span class="status-dot"></span>ChromaDB ready</span>
  <span><span class="status-dot"></span>Gemini-2.5-flash</span>
  <span><span class="status-dot"></span>Cache enabled</span>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("from_cache"):
            st.markdown('<span class="cache-badge">⚡ cached</span>', unsafe_allow_html=True)
        st.markdown(message["content"], unsafe_allow_html=True)

if query := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Peek at cache first so we know whether to show the badge
        is_cached = get_cached_answer(query) is not None

        if is_cached:
            st.markdown('<span class="cache-badge">⚡ cached</span>', unsafe_allow_html=True)
            response = run_agent(query)
        else:
            with st.spinner("Fetching sources and generating answer..."):
                response = run_agent(query)

        st.markdown(response, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "from_cache": is_cached
    })
