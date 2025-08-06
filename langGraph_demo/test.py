
import time
from openai import OpenAI

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

# 遍历每个模型进行测试
for config in model_configs:
    print(f"\n========== 正在测试模型：{config['name']} ==========")
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )

    start_time = time.time()
    first_token_time = None
    token_count = 0

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config["model"],
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
    print_result(config["name"], token_count, start_time, first_token_time, end_time)