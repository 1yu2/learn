# 01 项目搭建：dbt 是什么、项目结构、核心命令

## 为什么需要 dbt

数据团队常见的痛：数仓里堆了几百个 SQL 脚本，互相依赖没人说得清，改一个脚本不知道会弄坏哪张报表，更没有测试和文档。

dbt（data build tool）解决的就是这个「转换层的工程化」问题。它只管 **ELT 中的 T**：

- 数据已经由别的工具（Fivetran、Airbyte、手动脚本……）**加载（E、L）**进数仓了；
- dbt 负责在数仓**内部**把原始数据**转换（T）**成干净的、可分析的表。

dbt **不抽数、不搬数**，你写的每个 model 最终都会被编译成普通 SQL 发给数仓执行。所以学 dbt 的前提是 SQL，dbt 本身只是「组织 SQL 的框架」。

## dbt Core vs dbt Cloud

| | dbt Core | dbt Cloud |
| --- | --- | --- |
| 形态 | 开源 Python 命令行工具 | 官方托管的商业 SaaS |
| 运行环境 | 自己电脑 / 自己的服务器 | 浏览器 + 云端调度 |
| 调度、Web IDE、CI 托管 | 没有（自己配 cron / Airflow） | 内置 |
| 费用 | 免费 | 按席位收费 |

学习阶段用 **dbt Core** 就够了，语法和概念两边完全通用。本系列笔记和 `practice/` 项目都用 dbt Core + DuckDB（本地嵌入式数仓，零成本）。

安装（在隔离虚拟环境里，别污染系统 Python）：

```bash
uv venv && source .venv/bin/activate
uv pip install dbt-duckdb   # dbt-core + DuckDB 适配器
```

## 项目结构

一个 dbt 项目就是一个目录，最小骨架：

```
practice/
├── dbt_project.yml     # 项目配置（必需，项目根的标志）
├── profiles.yml        # 连接配置（默认在 ~/.dbt/，也可放项目里）
├── models/             # 转换逻辑：一个 .sql 文件 = 一个 model
├── seeds/              # CSV 文件，dbt seed 加载进数仓
├── tests/              # singular tests（自定义 SQL 测试）
├── macros/             # Jinja 宏（可复用 SQL 片段）
└── target/             # 编译产物与运行日志（自动生成，别手改）
```

### dbt_project.yml 常用字段

```yaml
name: "support_kpi"        # 项目名，必须是合法标识符；yml 里配置 models 时按这个名字分层
version: "1.0.0"

profile: "support_kpi"     # 用 profiles.yml 里哪个 profile 连接数仓

model-paths: ["models"]    # 各类资源目录，都可自定义
seed-paths: ["seeds"]
test-paths: ["tests"]
macro-paths: ["macros"]

target-path: "target"      # 编译产物目录
clean-targets:             # dbt clean 会删掉的目录
  - "target"
  - "dbt_packages"

models:                    # 按目录层级给 model 统一配置（+ 前缀表示配置项）
  support_kpi:             # ← 项目名
    staging:
      +materialized: view  # staging 层全部建成 view
      +tags: ["staging"]   # 统一打 tag，配合 --select tag:staging 使用
    marts:
      +materialized: table # marts 层建成 table
```

> 对应 omnisupport-copilot：`analytics/dbt_project.yml` 就是这个结构——staging/intermediate 用 view，marts 用 table，并统一打了 `week05` tag。

### profiles.yml 与 ~/.dbt

`profiles.yml` 是**连接数仓的凭证配置**，和项目逻辑分离（所以默认放在 `~/.dbt/profiles.yml`，避免把密码提交进 Git）。结构：profile 名 → `outputs`（一套环境一套连接，如 dev/prod）→ `target`（默认用哪套）。

```yaml
support_kpi:               # 必须和 dbt_project.yml 的 profile 字段一致
  target: dev
  outputs:
    dev:
      type: duckdb
      path: dev.duckdb
      threads: 4
```

如果像 practice 项目一样把 `profiles.yml` 放在项目目录里，运行时要加 `--profiles-dir .`（或设环境变量 `DBT_PROFILES_DIR`）。omnisupport-copilot 的 `analytics/profiles.yml.example` 是 Postgres 版本——换适配器只改这个文件，model SQL 基本不动，这正是 profiles 存在的意义。

## 核心命令

| 命令 | 干什么 |
| --- | --- |
| `dbt debug` | 体检：检查 dbt_project.yml / profiles.yml 是否合法、能否连上数仓。环境出问题先跑它 |
| `dbt run` | 编译并执行 models/ 下的所有 model（建 view/table） |
| `dbt test` | 跑所有测试（yml 里的 generic + tests/ 里的 singular） |
| `dbt build` | **一把梭**：按依赖顺序跑 seeds → models → tests，任一失败就停下游。日常主力命令 |
| `dbt seed` | 把 seeds/ 下的 CSV 加载进数仓成表 |
| `dbt docs generate` | 生成文档站点数据（manifest.json + catalog.json） |
| `dbt docs serve` | 起本地 Web 文档站，能看血缘图 |
| `dbt deps` | 安装 packages.yml 里声明的第三方包（如 dbt_utils） |
| `dbt clean` | 删除 target/ 和 dbt_packages/ |
| `dbt parse` | 只解析项目不执行，检查语法/Jinja 错误，速度最快 |

## node selection（节点选择）基础

项目一大，不可能每次全跑。`--select`（简写 `-s`）按条件挑节点：

```bash
dbt run --select stg_tickets          # 单个 model
dbt run --select staging              # models/staging/ 目录下全部
dbt build --select tag:marts          # 带 marts 标签的所有节点
dbt build --select stg_tickets+       # stg_tickets 及其所有下游（+ 在右边）
dbt build --select +support_kpi_mart  # 及其所有上游（+ 在左边）
dbt run --select tag:kpi tag:daily    # 多个条件 = 并集
```

`--exclude` 用法相同，表示排除。tag 在 `dbt_project.yml` 或 model 的 config 里打。practice 项目里可以试：

```bash
dbt build --select tag:marts --profiles-dir .
```

## L4 自检

不看文档，能完成以下任务才算真懂：

- [ ] 用 30 秒向同事解释 dbt 管 ELT 的哪一段、不管哪一段。
- [ ] 从零创建一个能 `dbt debug` 通过的项目骨架（手写 dbt_project.yml 和 profiles.yml，不用 `dbt init`）。
- [ ] 说出 profiles.yml 默认放哪、为什么要和项目分离、放项目里时怎么让 dbt 找到它。
- [ ] 说出 `dbt run`、`dbt test`、`dbt build` 三者的区别，以及为什么日常用 build。
- [ ] 在 practice 项目里：只跑 marts 层（用 tag）；只跑 `stg_tickets` 和它的全部下游。
- [ ] target/ 目录是干什么的？删掉会怎样？（用 `dbt clean` 验证你的答案。）
