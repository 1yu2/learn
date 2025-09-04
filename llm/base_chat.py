from langchain.chat_models import init_chat_model
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from config import LangFuseSession,model_configs,langfuse_configs

langfuse = Langfuse(
    **langfuse_configs
)

langfuse_handler = CallbackHandler()


model = init_chat_model(
    base_url=model_configs['base_url'],
    api_key=model_configs['api_key'],
    temperature=model_configs['temperature']
)

# 配置不同的sessionid
# import datetime

# LangFuseSession.session_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")