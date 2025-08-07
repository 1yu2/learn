
import time
from openai import OpenAI
import sys
import os
import requests

# 设置环境变量，确保不走代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['NO_PROXY'] = '*'

# 备选方案：使用requests库设置代理
# session = requests.Session()
# session.proxies = {
#     'http': None,
#     'https': None
# }

# Add the parent directory to the Python path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 测试 prompt 内容
prompt = "生成一张波斯猫的图片"

from config import model_configs

# 统一输出格式
def print_result(name, token_count, start_time, first_token_time, end_time):
    print(f"\n🧪 模型名称：{name}")
    if first_token_time:
        print(f"⏱ 首 token 响应时间：{first_token_time - start_time:.2f} 秒")
    print(f"⏱ 总耗时：{end_time - start_time:.2f} 秒")
    if first_token_time:
        duration = end_time - first_token_time
        print(f"📈 平均吞吐率：{token_count / duration:.2f} tokens/sec")
    print("-" * 40)

# 使用单个模型配置进行测试
print(f"\n========== 正在测试模型：{model_configs['model_name']} ==========")
client = OpenAI(
    base_url=model_configs["base_url"],
    api_key=model_configs["api_key"],
    http_client=None,  # 不使用代理
)

start_time = time.time()
first_token_time = None
token_count = 0

response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model=model_configs["model_name"],
    temperature=1.0,
    stream=True,
    timeout=60
)

for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
        token_count += 1
        if first_token_time is None:
            first_token_time = time.time()

end_time = time.time()
print_result(model_configs["model_name"], token_count, start_time, first_token_time, end_time)