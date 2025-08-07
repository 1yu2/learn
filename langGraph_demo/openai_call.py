from typing import List, Dict, Any
import os

# 设置环境变量，确保不走代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'

from config import model_configs
import openai
from openai import OpenAI
import json


def chat_client(model_name: str, base_url=None):
    """
    Create and return an OpenAI client for chat completions.
    The model_name will be used when making chat completion requests.
    """
    client = OpenAI(api_key=model_configs["api_key"], base_url=base_url)
    return client


def openai_api_call(messages, user_id=None):
    """
    智能API调用，能处理文本prompt字符串或复杂messages列表。
    """
    try:
        # 打印请求信息用于调试
        print("\n=== API Request Debug Info ===")
        print(f"Model: {model_configs['model_name']}")
        print(f"Base URL: {model_configs['base_url']}")
        
        # 创建一个用于打印的消息副本，避免打印大量base64数据
        debug_messages = []
        for msg in messages:
            debug_msg = msg.copy()
            if isinstance(debug_msg.get('content'), list):
                debug_content = []
                for item in debug_msg['content']:
                    if item.get('type') == 'image_url':
                        debug_content.append({
                            'type': 'image_url',
                            'image_url': {'url': '[BASE64_IMAGE_DATA]'}
                        })
                    else:
                        debug_content.append(item)
                debug_msg['content'] = debug_content
            debug_messages.append(debug_msg)
        
        print("\nRequest Messages Structure:")
        print(json.dumps(debug_messages, indent=2, ensure_ascii=False))

        client = OpenAI(
            base_url=model_configs["base_url"],
            api_key=model_configs["api_key"],
        )

        # 构建请求数据
        request_data = {
            "messages": messages,
            "model": model_configs["model_name"]
        }

        response = client.chat.completions.create(**request_data)
        return response

    except Exception as e:
        print("\n=== API Error Details ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        raise  # 重新抛出异常，保持原有的错误处理逻辑
