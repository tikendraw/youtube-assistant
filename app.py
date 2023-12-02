import logging

import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama

from main import extract_youtube_video_id, get_answer, get_youtube_video_to_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='streamlit-youtube-assistant.log')
logger = logging.getLogger(__name__)



model = 'mistral'

@st.cache_resource
def get_model_and_embedding(model_name):
    embeddings = OllamaEmbeddings(model=model_name)
    llm = Ollama(
        model=model,
        temperature=.2,
        repeat_penalty=1.3,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])

    )
    return llm, embeddings

def main():
    st.title(':robot: Youtube-assistant') 
    st.header("Youtube Assistant")
    st.text("""Video is too long? Zoned out? Ask it here!""")

    youtube_url = st.text_input('Youtube url here', placeholder='https://www.youtube.com/watch?v=XXXXXXXXXXX')
    query = st.text_input('Your curiosity...', placeholder='What is that...') 
    
    submit = st.button('Answer Me.')
    llm, embeddings = get_model_and_embedding(model_name=model)

    if youtube_url and query and submit:

        
        try:
            video_id = extract_youtube_video_id(youtube_url)
        except ValueError as e:
            st.error(e)
            
        try:
            db = get_youtube_video_to_db(video_id, embeddings)
        except Exception as e:
            st.text(e)
            return    
        
        st.markdown(f"# {query}")
        with st.spinner():
            result = get_answer(query, llm, db, k = 5)
            
        logger.info(f'Result for query "{query}": {result}')
        
        st.markdown(result)
        

if __name__=='__main__':
    main()