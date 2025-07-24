import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Streamlit session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def generate_text(prompt, max_length=200, model="meta-llama/llama-3.1-8b-instruct"):
    """
    Generates text using OpenRouter's API with streaming support.
    """
    try:
        # Initialize OpenAI client for OpenRouter
        client = OpenAI(
            api_key=st.session_state.api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # Make the API call with streaming
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise, accurate, and conversational answers to questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length,
            temperature=0.7,
            stream=True
        )

        # Stream the response
        generated_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                generated_text += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content  # Yield chunks for real-time display
        return generated_text.strip()
    except Exception as e:
        st.error(f"Error generating text: {str(e)}")
        return ""

def main():
    st.set_page_config(page_title="Question-Answering Chatbot", page_icon="💬", layout="centered")
    st.title("Question-Answering Chatbot")
    st.markdown("Ask any question, and get concise, conversational answers powered by OpenRouter.")

    # API Key input
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
        if api_key:
            st.session_state.api_key = api_key
        else:
            st.warning("Please enter your OpenRouter API key.")

    # Chat interface
    st.subheader("Chat")
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        if not hasattr(st.session_state, "api_key") or not st.session_state.api_key:
            st.error("Please enter your OpenRouter API key in the sidebar.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Generate and stream response
        with chat_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in generate_text(user_input, max_length=200):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")  # Cursor for typing effect
                response_placeholder.markdown(full_response)  # Final response without cursor

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()