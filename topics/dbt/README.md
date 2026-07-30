# dbt (data build tool) 学习

数据转换/分析工程工具，ELT 中的 T。以官方文档和官方课程为准。

## 官方资源

| 资源 | 链接 | 用途 |
| --- | --- | --- |
| 官方文档 | https://docs.getdbt.com/ | 主参考，所有功能权威说明 |
| 快速上手 | https://docs.getdbt.com/docs/get-started-dbt | 入门第一步 |
| dbt Learn 官方课程 | https://learn.getdbt.com/ （[Fundamentals](https://learn.getdbt.com/courses/dbt-fundamentals)） | 免费视频课，带实验 |
| Best Practices | https://docs.getdbt.com/best-practices | 官方项目组织规范 |
| dbt Style Guide | https://github.com/dbt-labs/corp/blob/main/dbt_style_guide.md | SQL 风格规范 |
| Package Hub | https://hub.getdbt.com/ | 社区包（dbt_utils 等） |
| dbt-core 源码 | https://github.com/dbt-labs/dbt-core | 进阶：读源码 |
| 社区 | https://discourse.getdbt.com/ · [Slack](https://www.getdbt.com/community/join-the-community/) | 提问 |

## 学习环境

推荐 **dbt Core + DuckDB**：零成本、纯本地、无需云数仓账号。

```bash
uv venv && uv pip install dbt-duckdb
dbt init <project_name>   # 适配器选 duckdb
```

练习项目放 `topics/dbt/practice/`（`dbt init` 生成的项目）。

## 学习路径与进度

### 阶段 1：核心基础（必会）

| # | 主题 | 官方文档 | 状态 |
| --- | --- | --- | --- |
| 1 | 快速上手 + 项目结构（dbt_project.yml、profiles.yml） | [get-started](https://docs.getdbt.com/docs/get-started-dbt) | ⬜ |
| 2 | Models（物化方式：view/table/incremental） | [models](https://docs.getdbt.com/docs/build/models) | ⬜ |
| 3 | ref / source 与 DAG | [sources](https://docs.getdbt.com/docs/build/sources) | ⬜ |
| 4 | Tests（generic + singular） | [tests](https://docs.getdbt.com/docs/build/data-tests) | ⬜ |
| 5 | Documentation 与血缘图 | [documentation](https://docs.getdbt.com/docs/build/documentation) | ⬜ |
| 6 | Jinja 与 Macros | [jinja-macros](https://docs.getdbt.com/docs/build/jinja-macros) | ⬜ |
| 7 | Seeds / Snapshots（SCD） | [seeds](https://docs.getdbt.com/docs/build/seeds) · [snapshots](https://docs.getdbt.com/docs/build/snapshots) | ⬜ |

### 阶段 2：工程化

| # | 主题 | 官方文档 | 状态 |
| --- | --- | --- | --- |
| 8 | 项目结构最佳实践（staging/marts 分层） | [best-practices](https://docs.getdbt.com/best-practices/how-we-structure/1-guide-overview) | ⬜ |
| 9 | 环境（dev/prod）、targets、变量 | [environments](https://docs.getdbt.com/docs/deploy/environments) | ⬜ |
| 10 | 常用包（dbt_utils、dbt_expectations） | [hub](https://hub.getdbt.com/) | ⬜ |
| 11 | 部署与调度（dbt build、CI、slim CI） | [deploy](https://docs.getdbt.com/docs/deploy/deployments) | ⬜ |
| 12 | Incremental 策略深入 + 性能优化 | [incremental](https://docs.getdbt.com/docs/build/incremental-models) | ⬜ |

### 阶段 3：进阶（选学）

| # | 主题 | 官方文档 | 状态 |
| --- | --- | --- | --- |
| 13 | Semantic Layer / MetricFlow | [semantic-layer](https://docs.getdbt.com/docs/use-dbt-semantic-layer/dbt-sl) | ⬜ |
| 14 | Exposures、Analyses、Hooks | [exposures](https://docs.getdbt.com/docs/build/exposures) | ⬜ |
| 15 | dbt Mesh / 跨项目引用 | [mesh](https://docs.getdbt.com/best-practices/how-we-mesh/mesh-1-intro) | ⬜ |
| 16 | dbt Fusion（新一代引擎） | [fusion](https://github.com/dbt-labs/dbt-fusion) | ⬜ |

配套课程：dbt Learn Fundamentals（免费，覆盖阶段 1）⬜

## 笔记

写在 `topics/dbt/notes/`（如 `02-models-materializations.md`）。按 [学习方法论](../../docs/learning-methodology.md)：dbt 是工具，核对"真懂"以 L4 为主——每个主题学完在 practice 项目里**不看文档写出来并跑通**。
