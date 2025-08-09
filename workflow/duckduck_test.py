from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults


## =========使用DuckDuckGoSearchRun进行一次性搜索=========
# 创建搜索对象
# search = DuckDuckGoSearchRun()

# # 执行搜索
# result = search.invoke("Obama's first name?")
# print(result)

## =========使用DuckDuckGoSearchResults获取详细搜索结果=========
# from langchain_community.tools import DuckDuckGoSearchResults

# # 创建搜索对象
# search = DuckDuckGoSearchResults()

# # 执行搜索
# results = search.invoke("Obama")
# for result in results:
#     print(result)


## ==================获取新闻搜索结果==================
# 创建新闻搜索对象
news_search = DuckDuckGoSearchResults(backend="news")

# 执行新闻搜索
news_results = news_search.invoke("Obama")
for news in news_results:
    print(news)

## ==================自定义搜索API包装器==================
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# # 定制搜索API包装器
# wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)

# # 使用定制的包装器进行搜索
# search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

# # 执行搜索
# custom_results = search.invoke("Obama")
# for result in custom_results:
#     print(result)
