import streamlit as st
from chat_backend import chat_with_agent, init_wiki_agent, get_wiki_agent_answer
from streamlit.components.v1 import html

import os
import streamlit as st
from PIL import Image
import html
import uuid

path = os.path.dirname(__file__)


icon_img = Image.open(os.path.join(path, "logo.png"))

USER_NAME = "Me"
AGENT_NAME = "Helpbot"


st.set_page_config(
    page_title="ChatGLM",
    page_icon=icon_img,
    layout="wide",
    # initial_sidebar_state="collapsed",
)

st.write(
    "<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True
)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


def javascript(source: str) -> None:
    """loading javascript correctly"""
    div_id = uuid.uuid4()

    st.markdown(
        f"""
    <div style="display:none" id="{div_id}">
        <iframe src="javascript: \
            var script = document.createElement('script'); \
            script.type = 'text/javascript'; \
            script.text = {html.escape(repr(source))}; \
            var div = window.parent.document.getElementById('{div_id}'); \
            div.appendChild(script); \
            div.parentElement.parentElement.parentElement.style.display = 'none'; \
        "/>
    </div>
    """,
        unsafe_allow_html=True,
    )


local_css("chat_style.css")


st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Josefin+Slab&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# User Input and Send button
user_profile_image = "https://img1.baidu.com/it/u=3150659458,3834452201&fm=253&fmt=auto&app=138&f=JPEG?w=369&h=378"

chatgpt_profile_image = (
    "https://cdn.dribbble.com/users/722835/screenshots/4082720/bot_icon.gif"
)


def display_chat_log(cur_container):
    for cur_conversation in st.session_state["chat_log"]:
        for msg in cur_conversation:
            if msg["role"] == USER_NAME:
                cur_container.markdown(
                    "<div class='   conversation-container user'><img src='{}' class='image-left' width='50'><br> {} </div>".format(
                        user_profile_image, html.escape(msg["content"])
                    ),
                    unsafe_allow_html=True,
                )
            else:
                cur_container.markdown(
                    "<div class='conversation-container bot'><img src='{}' class='image-left' width='50'></div>".format(
                        chatgpt_profile_image
                    ),
                    unsafe_allow_html=True,
                )
                cur_container.markdown(
                    f"{'&nbsp;&nbsp;&nbsp;'+msg['content']}", unsafe_allow_html=True
                )


def dict_to_github_markdown(data, has_section=False):
    wiki_logo = "https://upload.wikimedia.org/wikipedia/en/thumb/8/80/Wikipedia-logo-v2.svg/1200px-Wikipedia-logo-v2.svg.png"
    slack_logo = (
        "https://cdn.freebiesupply.com/logos/large/2x/slack-1-logo-png-transparent.png"
    )
    book_logo = "https://cdn-icons-png.flaticon.com/512/182/182956.png"
    markdown = ""
    for item in data:
        if "url" in item:
            title = item["title"]
            url = item["url"]
            if has_section:
                section = item["section"]
                title_text_and_section = f"{title} - {section}"
            else:
                title_text_and_section = title
            if "wikipedia" in url:
                logo = wiki_logo
            elif "slack" in url:
                logo = slack_logo
            else:
                logo = None
            if len(title_text_and_section) > 50:
                title_text_and_section = title_text_and_section[:50] + "..."
            hyperlink = f"[{title_text_and_section}]({url})"
            if logo:
                markdown += f"&nbsp;&nbsp;  <img src='{logo}' width='20' height='20'> {hyperlink} "
            else:
                markdown += f"&nbsp;&nbsp;  {hyperlink}"
        elif "chapter" in item:
            section = item["section"]
            if "chapter" in item and "page" in item:
                chapter = item["chapter"]
                if "总排放量" in chapter:
                    chapter = chapter.split("总排放量")[0]  # for better display
                page = item["page"]
                title_text_and_section = f"{chapter} (p. {page})"
            elif "page" in item:
                page = item["page"]
                title_text_and_section = f"p. {page}"
            elif "chapter" in item:
                chapter = item["chapter"]
                title_text_and_section = chapter
            else:
                continue
            markdown += f"&nbsp;&nbsp;  <img src='{book_logo}' width='20' height='20' title='Book' alt='Book'> {title_text_and_section} "
    return markdown


if "bot_desc" not in st.session_state:
    st.session_state["bot_desc"] = "General conversational Chatbot based on ChatGLM"


def clean_agent():
    st.session_state["chat_log"] = [[]]
    st.session_state["messages"] = None
    st.session_state["agent"] = None
    st.session_state["agent_chat_history"] = []
    if "agent_selected_str" in st.session_state:
        cur_agent = st.session_state["agent_selected_str"]
        # Set description
        bot_description = {
            "Chat": "General conversational Chatbot based on ChatGLM",
            "AI Wikipedia Agent": "Chat with knowlegebase. (OpenAI related wikipedia pages).",
            "Climate Book Agent": "Chat with Book: How to Avoid a Climate Disaster",
        }

        st.session_state["bot_desc"] = bot_description[cur_agent]


# Sidebar
st.sidebar.subheader("Model Settings")
agent_selected = st.sidebar.selectbox(
    label="Agent",
    options=["Chat", "AI Wikipedia Agent", "Climate Book Agent"],
    index=0,
    on_change=clean_agent,
    key="agent_selected_str",
    help="""Select the agent to chat with.\n\n
Chat: General conversational Chatbot based on ChatGLM.\n\n
AI Wikipedia Agent: Chat with knowlegebase. \n(OpenAI related wikipedia pages).
Climate Book Agent: Chat with Bill Gate's Book: How to Avoid a Climate Disaster
""",
)
max_token_selected = st.sidebar.slider(
    label="Model Max Output Length",
    min_value=50,
    max_value=4500,
    value=500,
    step=50,
    help="The maximum number of tokens to generate. Requests can use up to 2,048 or 4,000 tokens shared between prompt and completion. The exact limit varies by model. (One token is roughly 4 characters for normal English text)",
)
tempature_selected = st.sidebar.number_input(
    label="Model Tempature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1,
    help="Controls randomness: Lowering results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive.",
)

# Dynamic conversation display
if "chat_log" not in st.session_state:
    st.session_state["chat_log"] = [[]]

if "messages" not in st.session_state:
    st.session_state["messages"] = None
if "agent_chat_history" not in st.session_state:
    st.session_state["agent_chat_history"] = []

if "agent" not in st.session_state:
    st.session_state["agent"] = None


with st.form(key="user_question", clear_on_submit=True):

    # Title and Image in same line
    # Use user chatgpt profile image
    c1, c2 = st.columns((9, 1))
    c1.write("# ChatGLM")
    c1.write(f"### {st.session_state['bot_desc']}")

    help_bot_icon = (
        f'<img src="{chatgpt_profile_image}" width="60" style="vertical-align:middle">'
    )
    app_log_image = Image.open("logo.png")
    c2.image(app_log_image)

    conversation_main_container = st.container()

    user_input = st.text_area(
        "", key="user_input", height=20, placeholder="Ask me anything!"
    )
    # set button on the right
    _, c_clean_btn, c_btn, _ = st.columns([5.2, 1, 1.8, 2])
    send_button = c_btn.form_submit_button(label="Send")
    clean_button = c_clean_btn.form_submit_button(label="Clear")
    if clean_button:
        clean_agent()

    conversation = []
    if send_button:
        if user_input:
            with st.spinner("Thinking..."):
                # Determin which agent to call:
                if agent_selected == "Chat":

                    output, cur_chat_history = chat_with_agent(
                        user_input,
                        temperature=tempature_selected,
                        max_tokens=max_token_selected,
                        chat_history=st.session_state["messages"],
                    )

                    # Update chat history
                    st.session_state["messages"] = cur_chat_history
                    # Update overall displayed conversations
                    conversation.append({"role": USER_NAME, "content": user_input})
                    conversation.append({"role": AGENT_NAME, "content": output})
                elif agent_selected == "AI Wikipedia Agent":
                    if (
                        "agent" not in st.session_state
                        or st.session_state.agent is None
                    ):
                        st.session_state.agent = init_wiki_agent(
                            index_dir="index/openai_wiki_chinese_index_2023_03_24",
                            max_token=max_token_selected,
                            temperature=tempature_selected,
                        )
                    output_dict = get_wiki_agent_answer(
                        user_input,
                        st.session_state.agent,
                        chat_history=st.session_state["agent_chat_history"],
                    )
                    output = output_dict["answer"]

                    output_sources = [
                        c.metadata for c in list(output_dict["source_documents"])
                    ]

                    st.session_state["agent_chat_history"].append((user_input, output))

                    conversation.append({"role": USER_NAME, "content": user_input})
                    conversation.append(
                        {
                            "role": AGENT_NAME,
                            "content": output
                            + "\n\n&nbsp;&nbsp;&nbsp;**Sources:** "
                            + dict_to_github_markdown(output_sources, has_section=True),
                        }
                    )
                elif agent_selected == "Climate Book Agent":
                    if (
                        "agent" not in st.session_state
                        or st.session_state.agent is None
                    ):
                        st.session_state.agent = init_wiki_agent(
                            index_dir="index/how_to_avoid_climate_change_chinese_vectorstore",
                            max_token=max_token_selected,
                            temperature=tempature_selected,
                        )
                    output_dict = get_wiki_agent_answer(
                        user_input,
                        st.session_state.agent,
                        chat_history=st.session_state["agent_chat_history"],
                    )
                    output = output_dict["answer"]

                    output_sources = [
                        c.metadata for c in list(output_dict["source_documents"])
                    ]

                    st.session_state["agent_chat_history"].append((user_input, output))

                    conversation.append({"role": USER_NAME, "content": user_input})
                    conversation.append(
                        {
                            "role": AGENT_NAME,
                            "content": output
                            + "\n\n&nbsp;&nbsp;&nbsp;**Sources:** "
                            + dict_to_github_markdown(output_sources, has_section=True),
                        }
                    )

                st.session_state["chat_log"].append(conversation)
        col99, col1 = st.columns([999, 1])
        with col99:
            display_chat_log(conversation_main_container)
        with col1:

            # Scroll to bottom of conversation
            scroll_to_element = """
var element = document.getElementsByClassName('conversation-container')[
document.getElementsByClassName('conversation-container').length - 1
];
element.scrollIntoView({behavior: 'smooth', block: 'start'});
            """
            javascript(scroll_to_element)


def footer():
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
  """

    myargs = [
        "Made with ChatGLM models, check out the models on ",
        '<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="18" height="18" margin="0em">',
        ' <a href="https://github.com/THUDM/ChatGLM-6B" target="_blank">official repo</a>',
        "!",
        "<br>",
    ]

    st.markdown(style, unsafe_allow_html=True)
    st.markdown(
        '<div style="left: 0; bottom: 0; margin: 0px 0px 0px 0px; width: 100%; text-align: center; height: 30px; opacity: 0.8;">'
        + "".join(myargs)
        + "</div>",
        unsafe_allow_html=True,
    )


footer()
