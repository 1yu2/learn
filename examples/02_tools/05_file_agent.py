from datetime import datetime
from os import getenv
from pathlib import Path

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv
from httpx import Timeout

load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )

# 所有工具定义


def read_file(file_path: str) -> str:
    """
    读取文件内容

    Args:
        file_path (str): 文件路径

    Returns:
        str: 文件内容
    """
    try:
        path = Path(file_path).expanduser()
        if not path.exists():
            return f"错误：文件 {file_path} 不存在"
        if not path.is_file():
            return f"错误：路径 {file_path} 不是文件"

        content = path.read_text(encoding="utf-8")
        return f"成功读取文件，内容：\n{content}"
    except (OSError, UnicodeError) as error:
        return f"读取文件时出错：{error}"


def write_file(file_path: str, content: str) -> str:
    """
    写入内容到文件

    Args:
        file_path (str): 文件路径
        content (str): 要写入的内容

    Returns:
        str: 操作结果
    """
    try:
        path = Path(file_path).expanduser()
        path.write_text(content, encoding="utf-8")
        return f"成功将内容写入文件：{file_path}"
    except (OSError, UnicodeError) as error:
        return f"写入文件时出错：{error}"


def list_files(directory: str = ".") -> str:
    """
    列出目录下的所有文件

    Args:
        directory (str): 目录路径，默认为当前目录

    Returns:
        str: 文件列表
    """
    try:
        path = Path(directory).expanduser()
        if not path.exists():
            return f"错误：目录 {directory} 不存在"
        if not path.is_dir():
            return f"错误：路径 {directory} 不是目录"

        files = sorted(
            (entry for entry in path.iterdir() if entry.is_file()),
            key=lambda entry: (entry.name.casefold(), entry.name),
        )
        if not files:
            return f"目录 {directory} 下没有文件"

        file_list = "\n".join(f"- {file.name}" for file in files)
        return f"目录 {directory} 下的文件：\n{file_list}"
    except OSError as error:
        return f"列出文件时出错：{error}"


def get_file_info(file_path: str) -> str:
    """
    获取文件信息

    Args:
        file_path (str): 文件路径

    Returns:
        str: 文件信息
    """
    try:
        path = Path(file_path).expanduser()
        if not path.exists():
            return f"错误：文件 {file_path} 不存在"
        if not path.is_file():
            return f"错误：路径 {file_path} 不是文件"

        stat = path.stat()
        modified_time = datetime.fromtimestamp(stat.st_mtime)

        return (
            "文件信息：\n"
            f"- 路径：{file_path}\n"
            f"- 大小：{stat.st_size} 字节\n"
            f"- 修改时间：{modified_time:%Y-%m-%d %H:%M:%S}"
        )
    except OSError as error:
        return f"获取文件信息时出错：{error}"


# 创建Agent并绑定工具
file_agent = Agent(
    name="文件助手",
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    tools=[read_file, write_file, list_files, get_file_info],
    instructions=[
        "你是一个文件操作助手",
        "帮助用户读取、写入和管理文件",
        "操作前确认用户意图，避免误操作",
        "提供清晰的操作结果反馈",
    ],
    markdown=True,  # 使用Markdown格式输出
)

# 使用Agent
if __name__ == "__main__":
    # 示例：列出当前目录文件
    file_agent.print_response(
        "列出当前目录下的所有文件，并计算第三个文件的大小，"
        "然后读取文件内容后打印出来，最后关闭文件"
    )
