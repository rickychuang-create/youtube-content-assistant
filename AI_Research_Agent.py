import os
import re
import requests
import streamlit as st
import json
import concurrent.futures
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# ==============================
# 🔧 頁面設定 (Page Config)
# ==============================
st.set_page_config(
    page_title="YouTube 內容策略 AI 助理",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    /* 讓App標題與新的紅色主題色呼應 */
    h1, h2, h3 {color: #ff4b4b;} 

    /* 所有分頁的容器 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; /* 分頁之間的間距 */
        
        display: flex !important;
        width: 100% !important;
    }

    /* 未選中分頁的樣式 */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #444;
        transition: background-color 0.3s, color 0.3s;
        flex-grow: 1; /* 允許分頁伸展以填滿空間 */
        justify-content: center; /* 水平置中文字 */
        text-align: center; /* 確保文字居中對齊 */
    }

    /* 滑鼠懸停在「未選中」分頁上的樣式 */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover {
        background-color: #e8e8e8;
        color: #ff4b4b;
    }

    /* 已選中分頁的樣式 */
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)

# ==============================
# 🎨 UI 美化與標題
# ==============================
st.title("🤖 YouTube 內容策略 AI 助理")

# ==============================
# 🔗 連結驗證函式 (新增)
# ==============================
def check_url_status(url):
    """檢查單一 URL 的狀態，返回 True (有效) 或 False (無效)"""
    try:
        # 使用 HEAD 請求，速度更快，因為它只獲取標頭
        response = requests.head(url, allow_redirects=True, timeout=5)
        # 狀態碼在 200-399 之間都視為有效
        if 200 <= response.status_code < 400:
            return True
        else:
            return False
    except requests.RequestException:
        return False

# ==============================
# 🔑 API 金鑰與狀態初始化
# ==============================
# 讀取金鑰
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("錯誤：請先在 .streamlit/secrets.toml 中設定您的 OPENROUTER_API_KEY。")
    st.stop()

# --- 初始化 OpenAI Client ---
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)


# 初始化 session_state
if 'discovered_topics_text' not in st.session_state:
    st.session_state.discovered_topics_text = ""
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = None
if 'research_result' not in st.session_state:
    st.session_state.research_result = {}
if 'field_selection' not in st.session_state:
    st.session_state.field_selection = "自動探索當前美股熱門議題" # 預設值


# 儲存研究結果的檔案
storage_file = "research_results.json"

# ==============================
# 📂 資料存取函式
# ==============================
def load_stored_data():
    """從 JSON 檔案載入已儲存的研究結果"""
    if os.path.exists(storage_file):
        with open(storage_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_result(topic, perplexity_result):
    """將新的研究結果儲存到 JSON 檔案"""
    data = load_stored_data()
    data[topic] = {
        "perplexity_result": perplexity_result
    }
    with open(storage_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

stored_data = load_stored_data()

# ==============================
# 🧠 LLM API 呼叫函式
# ==============================
def openrouter_chat(model, messages, temperature=0.7):
    """使用 openai 函式庫呼叫 OpenRouter API"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=1200
        )
        content = completion.choices[0].message.content
        if content is None:
             st.error("API 回應格式錯誤：在回傳資料中找不到 'content'。")
             print("Received unexpected response structure:", completion)
             return None
        return content

    except APITimeoutError:
        st.error(f"API 請求超時：伺服器在 {600} 秒內未完成回應。")
        return None
    except RateLimitError as e:
        st.error(f"API 速率限制錯誤：請稍後再試。 ({e})")
        return None
    except APIError as e: # 捕捉更廣泛的 API 錯誤，例如伺服器錯誤 (5xx)
        st.error(f"OpenRouter API 錯誤 (HTTP {e.status_code})：{e.message}")
        print(f"API Error details: {e}")
        return None
    except Exception as e: # 捕捉其他潛在錯誤
        st.error(f"呼叫 API 時發生未預期的錯誤: {e}")
        print(f"Unexpected error details: {e}")
        return None

# ==============================
# 📝 Prompt 模板
# ==============================
DISCOVER_PROMPT_TEMPLATE = """
    # 指令：擔任財經內容策略師，為 YouTube 頻道發想「爆款」影片議題

    你是一位頂尖的財經內容策略師，嗅覺敏銳，專為專注於「美股、財經時事、投資理財」的 YouTube 頻道打造爆款內容。你的核心任務是**從社群的熱議中挖掘潛力話題，再用權威資訊加以驗證與深化**，最終產出一系列具備高觀看潛力、高討論度的影片議題。

    **研究領域：{field}**

    ### **核心方法論 (Core Methodology):**
    1.  **第一步：傾聽社群脈動。** 你的首要任務是深入 X (Twitter)、Reddit (例如 r/wallstreetbets, r/investing)、財經 Podcast 等社群平台，找出當前投資者群體中**最熱議、最具爭議、或被嚴重低估**的話題和觀點。
    2.  **第二步：權威驗證與深化。** 找到社群熱點後，再去尋找權威媒體和機構報告，目的是為這個熱點提供**數據支撐、專家背書，或是提出強烈的相反觀點**，從而創造內容的深度與衝突感。
    3.  **第三步：打造獨特觀點。** 結合社群的熱情與權威的數據，最終形成一個獨特、引人入勝的影片切入點。

    請嚴格遵循以上方法論和以下Markdown結構提供 3-5 個議題，並按照你最推薦的順序列出，完成你的議題推薦清單：

    ---

    ### **YouTube 影片議題策略清單**

    #### (建議影片標題，不要有符號，呈現標題就好)
    - 提供一個具有吸引力、符合 YouTube SEO、能激發好奇心的影片標題草案。

    **1. 核心內容摘要 (Content Summary):**
    - 簡要概述此議題將探討的核心問題、事件或概念。

    **2. 影片潛力評估 (Video Potential Assessment):**
    - **a. 核心衝突/看點 (Core Conflict/Hook):** 點出這個議題最吸引人的地方是什麼？是觀點的對立、驚人的數據、一個未解的謎團，還是一個與直覺相反的事實？這將是影片的敘事核心。
    - **b. 目標觀眾 (Target Audience):** 這個議題最能吸引哪一類觀眾？（例如：新手投資者、資深交易員、科技股愛好者、價值投資者等）。
    - **c. 視覺化潛力 (Visualization Potential):** 這個主題有哪些內容特別適合用視覺化方式呈現？（例如：股價走勢圖、公司營收結構圓餅圖、技術原理解說動畫、關鍵人物關係圖等）。
    - **d. 獨特切入點 (Unique Angle):** 目前市場上可能已經有相關影片，我們可以用什麼獨特的角度或觀點來做出差異化？

    **3. 關鍵資訊來源 (Key Information Sources):**
    - 列出 2-3 個最關鍵、最權威的起始資訊來源，**請務必確保你提供的 URL 是真實、有效且可訪問的，優先選擇來自大型、穩定的新聞網站、官方報告或權威機構的直接連結，**作為下一步【主題研究】的絕佳起點。

    ---
    ### **研究與引用指南 (Research & Citation Guidelines)**
    * **資訊來源的優先順序：** 你的研究必須基於以下來源，並**優先從第一類來源中尋找靈感**：
        1.  **社群平台與專家觀點 (社群熱點來源):** 來自 X (Twitter) 上的金融大 V、Reddit (如 r/wallstreetbets)、知名財經 Podcast 或 YouTube 頻道的熱門討論。
        2.  **權威媒體 (觀點深化來源):** 來自國際主流媒體（Bloomberg, Reuters）或頂尖行業媒體（TechCrunch）的深度報導，用於驗證或挑戰社群觀點。
        3.  **機構報告 (數據支撐來源):** 來自權威研究機構、頂級顧問公司的報告，主要用於為影片提供可信的數據和圖表。
    * **客觀性：** 即使從社群熱點出發，也需要客觀呈現不同角度的觀點。
    * **語言：** 請盡可能同時搜索並引用英文與中文的資料。
    * **時效性：** **社群議題請優先選用近3天的資料**，以確保話題熱度。
"""

RESEARCH_PROMPT_TEMPLATE = """
    # 指令：擔任專業研究員與內容策略師，生成一份影片腳本導向的分析報告

    你是一位頂尖的專業研究員，同時也是一位經驗豐富的內容策略師。你的任務是針對用戶提供的以下主題，進行全面、客觀的研究，並生成一份不僅內容扎實，且**極度有利於後續 YouTube 影片腳本撰寫**的結構化分析報告。

    **主題：{topic}**

    請嚴格遵循以下報告結構和研究指南來完成任務，且內容盡量提及重點並精簡，不需要有太多過於冗長的贅述，整個報告文字不超過6000字：

    ---

    ### **主題分析報告：{topic}**

    **1. 執行摘要 (Executive Summary)**
    - 用 2-3 句話簡潔有力地總結整個主題的核心內容。
    - 概述關於此主題最主要的幾種觀點或發現。
    - 提出基於研究的綜合性結論。

    **2. 主題背景與核心定義 (Background & Core Definition)**
    - 解釋此主題的背景、起源和發展脈絡。
    - 如果主題包含專有名詞，請給出清晰、易懂的定義。
    - 此段落文字不超過500字

    **3. 主要觀點與分析 (Key Perspectives & Analysis)**
    - **a. 正面觀點 / 支持方論點 / 機會 (Proponents' Views / Positive Developments / Opportunities):**
        - 總結支持此主題或對其持樂觀態度的主要論點。
        - 列舉其帶來的潛在好處、機會或正面影響。
        - 此段落文字不超過1000字
    - **b. 負面觀點 / 反對方論點 / 風險挑戰 (Critics' Views / Risks & Challenges):**
        - 總結反對此主題或對其持謹慎/悲觀態度的主要論點。
        - 列舉其可能帶來的風險、挑戰、爭議或負面影響。
        - 此段落文字不超過1000字
    - **c. 現況與關鍵案例 (Current Status & Key Cases):**
        - 描述此主題目前的發展現況。
        - 提供2個以上最相關的真實案例、數據或事件作為佐證。
        - 此段落文字不超過1000字

    **4. 未來展望 (Future Outlook)**
    - 基於以上分析，對此主題的未來發展趨勢做出預測。
    - 提出幾個值得未來持續關注的關鍵點。
    - 此段落文字不超過1000字

    **5. 總結 (Conclusion)**
    - 再次對整個研究進行簡要總結，重申最重要的發現。

    ---
    
    ### **研究與引用指南 (Research & Citation Guidelines)**
    * **資訊來源：** 你的研究必須基於以下三種類型的權威資訊。每種類型至少引用 3 個以上不同來源：
        1.  **機構報告：** 來自權威研究機構、頂級顧問公司、政府單位或國際組織的報告。
        2.  **權威媒體：** 來自國際主流媒體、頂尖行業媒體的深度報導或分析文章。
        3.  **專家觀點：** 來自該領域公認的專家、學者或行業領袖的公開言論（如學術論文、社群平台發文、演講、Podcast等）。
    * **客觀性：** 請保持客觀中立，全面呈現不同角度的觀點，避免帶有偏見的陳述。
    * **語言：** 請盡可能同時搜索並引用英文與中文的資料，以確保視角的全面性。
    * **時效性：** 請優先選用近 2 年內的資料，除非該主題的歷史背景至關重要。
    * **引用：** 所有關鍵論點、數據和直接引述都必須在報告結尾的「參考資料」部分清晰列出。

"""

SCRIPT_PROMPT_TEMPLATE = """
# 指令：擔任專業 YouTube 腳本寫手
你是一位頂尖的 YouTube 腳本寫手，擅長將複雜的研究報告轉化為吸引人的故事。

**核心任務：**
你的任務是將以下這份詳細的「主題分析報告」轉化為一份完整的、口語化的逐字稿腳本。

**關鍵指南：**
1.  **尋找大綱：** 請**優先尋找**報告中第 5 點「腳本撰寫輔S助元素」裡的 `f. 建議的影片敘事結構`。
2.  **遵循大綱：** **嚴格以此「敘事結構」（例如：幕一、幕二...）作為你的腳本骨架**，來組織整部影片的故事線。
3.  **填充內容：** 將報告中的其他元素（如鉤子、數據、案例、比喻）填充到這個敘事骨架的相應位置。

**影片主題：{topic}**
---
**研究報告全文：**
{final_summary}
---
**腳本要求：**
1.  **結構 (Structure):** 嚴格遵循報告中 `f. 建議的影片敘事結構`（例如 幕一, 幕二, 幕三...）來組織故事。
2.  **開場 (Hook):** 使用報告中建議的「引人入勝的切入點」作為影片的開頭。
3.  **內容 (Body):** 將報告中「主要觀點與分析」的複雜內容，用「故事化敘事元素」和「比喻」重新包裝，使其生動易懂。
4.  **視覺提示 (Visuals):** 在腳本中適當位置，用 `[畫面提示：...]` 插入報告中建議的「視覺化素材建議」。
5.  **結尾 (Outro):** 用「核心傳達理念」來總結觀點，並用「引發思考的問題」來引導觀眾留言互動。
6.  **語氣 (Tone):** 口語化、有觀點、像一個聰明的朋友在為您深入解析，但不會太嚴肅。
"""

# ==============================
# 🤖 Agent 核心功能函式
# ==============================
def discover_topics(field):
    prompt = DISCOVER_PROMPT_TEMPLATE.format(field=field)
    return openrouter_chat(
        model="perplexity/sonar-pro",
        messages=[{"role":"user","content":prompt}]
    )

def search_with_perplexity(topic):
    prompt = RESEARCH_PROMPT_TEMPLATE.format(topic=topic)
    return openrouter_chat(
        model="perplexity/sonar-deep-research",
        messages=[{"role": "user", "content": prompt}]
    )

def generate_video_script(topic, final_summary):
    prompt = SCRIPT_PROMPT_TEMPLATE.format(topic=topic, final_summary=final_summary)
    return openrouter_chat(
        model="anthropic/claude-3.5-sonnet", # 使用更擅長創意寫作的模型
        messages=[{"role":"user","content":prompt}],
        temperature=0.8
    )

def verify_links_in_text(text):
    """驗證文本中的連結並附加狀態圖示"""
    if not text: return ""
    
    urls = re.findall(r'https?://[^\s\)\>]+', text)
    if not urls:
        return text

    verified_text = text
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(check_url_status, url): url for url in set(urls)}
        url_statuses = {}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                is_valid = future.result()
                url_statuses[url] = "✅" if is_valid else "❌"
            except Exception:
                url_statuses[url] = "⚠️"

    for url, status in url_statuses.items():
        verified_text = verified_text.replace(url, f'{url} {status}')
    
    return verified_text

# ==============================
# 🖼️ Streamlit 介面佈局
# ==============================
# 創建三個分頁
tab1, tab2, tab3 = st.tabs(["**Step 1: 🕵️ 探索高潛力影片議題**", "**Step 2: 🔍 深度議題研究**", "**Step 3: 🎬 影片腳本**"])

# ------------------------------
# Tab 1：主題探索
# ------------------------------
with tab1:
    st.header("🕵️ 探索高潛力影片議題")
    st.markdown('''
    **核心方法論 (Core Methodology):** 
    - **第一步：傾聽社群脈動。** 深入 X (Twitter)、Reddit (例如 r/wallstreetbets, r/investing)、財經 Podcast 等社群平台，找出當前投資者群體中**最熱議、最具爭議、或被嚴重低估**的話題和觀點。
    - **第二步：權威驗證與深化。** 找到社群熱點後，再去尋找權威媒體和機構報告，目的是為這個熱點提供**數據支撐、專家背書，或是提出強烈的相反觀點**，從而創造內容的深度與衝突感。
    - **第三步：打造獨特觀點。** 結合社群的熱情與權威的數據，最終形成一個獨特、引人入勝的影片切入點。
    ---
    '''
    )

    PREDEFINED_FIELDS = [
        "自動探索當前美股熱門議題",
        "AI",
        "半導體",
        "電動車與新能源",
        "總體經濟 (利率、通膨)",
        "其他 (自行輸入)"
    ]
    AUTO_SEARCH_INSTRUCTION = "不限制特定領域，請自動從 X (Twitter)、Reddit 等社群，結合主流財經媒體，找出近期整個美股市場最熱門、最具討論度的潛力話題。"

    st.markdown("**請選擇您感興趣的領域：**")
    
    # --- 修正點 1：使用 st.session_state 記住下拉選單的選擇 ---
    # 這樣即使頁面重整，選擇也不會跑掉
    if 'field_selection' not in st.session_state:
        st.session_state.field_selection = PREDEFINED_FIELDS[0]

    # 將 selectbox 的選擇直接綁定到 session_state
    field_selection = st.selectbox(
        "領域選單", 
        PREDEFINED_FIELDS, 
        key='field_selection', # 綁定 key
        label_visibility="collapsed"
    )
    
    custom_field_input = ""
    # --- 修正點 2：根據 session_state 的值來決定是否顯示文字框 ---
    if st.session_state.field_selection == "其他 (自行輸入)":
        custom_field_input = st.text_input("請輸入您想研究的其他領域：", placeholder="例如：太空科技、生技醫療...")

    # --- 修正點 3：直接使用 st.button，不再需要 form ---
    if st.button("💡 生成熱門議題", type="primary"):
        final_field = ""
        if st.session_state.field_selection == "其他 (自行輸入)":
            if custom_field_input.strip():
                final_field = custom_field_input.strip()
            else:
                st.warning("您選擇了「其他」，但未輸入任何領域。")
        elif st.session_state.field_selection == "自動探索當前美股熱門議題":
            final_field = AUTO_SEARCH_INSTRUCTION
        else:
            final_field = st.session_state.field_selection

        if final_field:
            with st.spinner("🧠 正在掃描全球資訊..."):
                topics_text = discover_topics(final_field)
                if topics_text:
                    # 直接驗證連結並儲存完整文本
                    st.session_state.discovered_topics_text = verify_links_in_text(topics_text)
                    st.success("✅ 已為您生成推薦議題！")
                else:
                    st.session_state.topic_list = []
                    st.error("無法生成主題，請檢查 API 連線或稍後再試。")

    if st.session_state.discovered_topics_text:
        st.divider()
        st.subheader("推薦議題列表", anchor=False)
        st.caption("✅: 連結有效 | ❌: 連結無效或無法訪問 | ⚠️: 驗證時發生錯誤")
        with st.container(border=True):
             st.markdown(st.session_state.discovered_topics_text, unsafe_allow_html=True)




# ------------------------------
# Tab 2：主題研究
# ------------------------------
with tab2:
    st.header("🔍 深度議題研究")
    st.markdown("此步驟將使用 **Perplexity Deep Research** 模型針對以下您輸入的議題進行深入的網路資料探勘與分析，產出影片腳本導向的完整報告。")

    topic_input_value = st.session_state.selected_topic if st.session_state.selected_topic else ""
    topic_input = st.text_input("請輸入或確認要研究的主題：", value=topic_input_value)

    if st.button("🚀 開始深度研究", type="primary") and topic_input.strip():
        # 檢查快取
        if topic_input in stored_data:
            st.info("偵測到此主題的歷史研究紀錄，將直接從快取載入。")
            perplexity_result = stored_data[topic_input]["perplexity_result"]
            st.session_state.research_result[topic_input] = perplexity_result
        else:
            with st.spinner(f"🔍 正在為您進行「{topic_input}」的 Deep Research，過程可能需要 5-10 分鐘，請稍候..."):
                perplexity_result = search_with_perplexity(topic_input)
                if perplexity_result:
                    save_result(topic_input, perplexity_result)
                    st.session_state.research_result[topic_input] = perplexity_result
                    st.success("✅ 主題研究完成並已儲存！")
                else:
                    st.error("研究過程中發生錯誤，請稍後再試。")

    # 顯示研究結果
    if topic_input and topic_input in st.session_state.research_result:
        st.divider()
        st.subheader(f"📚 研究報告：{topic_input}", anchor=False)
        with st.container(border=True):
            st.markdown(st.session_state.research_result[topic_input])

# ------------------------------
# Tab 3：影片腳本
# ------------------------------
with tab3:
    st.header("🎬 生成 YouTube 影片腳本")
    st.info("AI 將化身專業腳本寫手，根據第二步產出的深度研究報告，為您撰寫一份生動有趣的口語化逐字稿。")

    # 讓使用者從已研究的主題中選擇
    researched_topics = list(stored_data.keys())
    if not researched_topics:
        st.warning("目前尚無已完成的研究報告。請先在「Step 2: 主題研究」分頁完成至少一項研究。")
    else:
        selected_topic_for_script = st.selectbox(
            "選擇一個已完成研究的主題來生成腳本：",
            options=researched_topics,
            index=None,
            placeholder="請選擇..."
        )

        if st.button("✍️ 生成影片腳本", type="primary") and selected_topic_for_script:
            final_summary = stored_data[selected_topic_for_script]["perplexity_result"]
            with st.spinner(f"🎬 您的專屬腳本寫手正在為「{selected_topic_for_script}」撰寫腳本..."):
                video_script = generate_video_script(selected_topic_for_script, final_summary)
                if video_script:
                    st.success("✅ 影片腳本生成完成！")
                    with st.container(border=True):
                        st.markdown(video_script)
                else:
                    st.error("腳本生成失敗，請稍後再試。")


