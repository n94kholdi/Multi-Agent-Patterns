import streamlit as st
from src.MAS_hierarichal.core import orchestrate_graph

class App:
    def __init__(self):
        st.set_page_config(page_title="Research and Write Hierarichal Agentic system", page_icon= "content/logo.png")
        st.title("🦜 Research and Write Hierarichal Agentic system")
        st.markdown(f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500&display=swap');
                * {{font-family: 'Vazirmatn', sans-serif;}}
            </style>
        """, unsafe_allow_html=True)

    def display_sidebar(self):
        api_key = st.sidebar.text_input("API Key", type="password")
        if not api_key:
            st.info("Please enter API key to continue.")
            st.stop()
        return api_key

    def display_chat(self, chat_history):
        if len(chat_history) == 0 or st.sidebar.button("Reset chat history"):
            return True 
        
        for msg in chat_history:
            if msg.type == 'AIMessageChunk':
                msg.type = 'ai'
            st.chat_message(msg.type).write(msg.content)
        return False

    def get_user_input(self):
        return st.chat_input()

    def display_message(self, message_type, content):
        if message_type == "human":
            st.chat_message(message_type).write(content)
        else:
            with st.chat_message("ai"):
                # st.write_stream(content)
                if isinstance(content, dict) and "output" in content:
                    st.write(content["output"])
                else:
                    st.write(content)

    def display_message_stream(self, sender, message, delay=0.05):
        import time

        st.write(f"{sender}: ", end="")  # initial label
        words = message.split(" ")
        for word in words:
            st.write(word + " ", end="")
            time.sleep(delay)
            
    def display_multi_messages(self, backend, prompt):

        for resp in backend.stream(
            {
                "messages":[
                    ("user", prompt)
                ]
            },
            {"recursion_limit": 150},
        ):
            print(resp)
            print("---")
            agent_team = list(resp.keys())[0]
            if agent_team != 'supervisor':      
                if isinstance(resp[agent_team], dict) and "messages" in resp[agent_team]:         
                    self.display_message("ai", resp[agent_team]["messages"][0].content)

    def display_app(self):
        api_key = self.display_sidebar()
        backend = orchestrate_graph.orchestrate_graph(api_key).create_graph()

        # if self.display_chat(backend.get_chat_history()):
        #     backend.reset_chat()

        prompt = self.get_user_input()
        if prompt:
            self.display_message("human", prompt)
            self.display_multi_messages(backend, prompt)

if __name__ == "__main__":
    app = App()
    app.display_app()