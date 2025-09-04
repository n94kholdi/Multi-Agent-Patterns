import streamlit as st
from src.agents.travel_agent import TravelAgent
from src.agents.supervisor import SupervisorAgent

class App:
    def __init__(self):
        st.set_page_config(page_title="Travel Planning Agent", page_icon= "content/logo.png")
        st.title("🦜 Travel Planning Agent")
        st.markdown(f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500&display=swap');
                * {{font-family: 'Vazirmatn', sans-serif;}}
            </style>
        """, unsafe_allow_html=True)

    def display_sidebar(self):
        api_key = st.sidebar.text_input("Cohere API Key", type="password")
        if not api_key:
            st.info("Please enter Cohere API key to continue.")
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

    def display_app(self):
        api_key = self.display_sidebar()
        backend = SupervisorAgent(
                                provider="cohere",
                                api_key=api_key,
                                streaming=True,
                                use_history=True)#.create_supervisor_agent()
        # backend = TravelAgent("cohere", api_key)

        # if self.display_chat(backend.get_chat_history()):
        #     backend.reset_chat()

        prompt = self.get_user_input()
        if prompt:
            self.display_message("human", prompt)
            response = backend.process_input(prompt)
            self.display_message("ai", response)

if __name__ == "__main__":
    app = App()
    app.display_app()