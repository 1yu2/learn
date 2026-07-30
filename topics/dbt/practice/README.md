# dbt Core + DuckDB 练习项目：Support KPI

面向零基础的 dbt 练习项目，主题为「客服工单支持指标」，结构参照
`/Volumes/Move/omnisupport-copilot/analytics/`（staging / intermediate / marts 三层 + tests + seeds）。

## 环境准备

```bash
cd topics/dbt/practice

# 已有 .venv（uv venv 创建，已装 dbt-duckdb）。如需重建：
# uv venv && uv pip install dbt-duckdb

source .venv/bin/activate
```

## 常用命令

本项目把 `profiles.yml` 放在项目目录里（而不是默认的 `~/.dbt/`），所以每个命令都要加 `--profiles-dir .`：

```bash
dbt debug --profiles-dir .                       # 检查连接与配置
dbt seed --profiles-dir .                        # 只加载 CSV 原始数据
dbt run --profiles-dir .                         # 只跑模型，不跑测试
dbt test --profiles-dir .                        # 只跑测试
dbt build --profiles-dir .                       # seeds + models + tests 一把梭（推荐）
dbt build --select tag:marts --profiles-dir .    # 只跑带 marts 标签的节点（演示 tag 选择）
dbt build --select stg_tickets --profiles-dir .  # 只跑单个模型
dbt docs generate --profiles-dir .               # 生成文档（target/manifest.json、catalog.json）
dbt docs serve --profiles-dir .                  # 起本地文档站，可看血缘图
dbt clean --profiles-dir .                       # 清掉 target/ 和 dbt_packages/
```

数据落在项目目录下的 `dev.duckdb`（DuckDB 本地文件）。可以直接查：

```bash
.venv/bin/python -c "
import duckdb
con = duckdb.connect('dev.duckdb', read_only=True)
print(con.sql('select * from support_kpi_mart').fetchall())"
```

## 文件结构

```
practice/
├── dbt_project.yml              # 项目配置：名称、路径、各层物化方式与 tag
├── profiles.yml                 # 连接配置：DuckDB 本地文件 dev.duckdb
├── seeds/                       # 原始数据（dbt seed 加载为数仓里的表）
│   ├── raw_tickets.csv          #   14 行工单，含边界数据：T009 是跨 40 天的超长工单、
│   │                            #   T003/T006/T011/T013 未解决（resolved_at 为空）
│   ├── raw_customers.csv
│   └── raw_ticket_comments.csv
├── models/
│   ├── staging/                 # staging 层（view）：清洗、类型转换、标准化
│   │   ├── sources.yml          #   声明 seed 表为 source（演示 source() 用法）
│   │   ├── stg_tickets.sql      #   用 {{ source('raw', 'raw_tickets') }} 取数
│   │   ├── stg_customers.sql
│   │   ├── stg_ticket_comments.sql
│   │   └── schema.yml           #   description + generic tests（unique/not_null/accepted_values/relationships）
│   ├── intermediate/            # intermediate 层（view）：业务加工
│   │   ├── int_ticket_activity_daily.sql  # is_p1 / sla_breached / resolution_hours，按天+类别聚合
│   │   └── schema.yml
│   └── marts/                   # marts 层（table）：对外交付
│       ├── support_kpi_mart.sql           # 按类别的 KPI：工单数、解决时长、SLA 达标率
│       └── schema.yml
└── tests/                       # singular tests（写业务规则的 SQL，返回 0 行 = 通过）
    ├── kpi_counts_non_negative.sql        # 计数指标不允许为负
    └── sla_compliance_rate_range.sql      # SLA 达标率必须在 [0,1]
```

## 与 omnisupport-copilot 的对应

| 本项目 | omnisupport-copilot/analytics/ |
| --- | --- |
| `seeds/*.csv` | PostgreSQL 里的原始表（`sources.yml` 中 `omni_postgres` 声明的那批） |
| `models/staging/stg_tickets.sql` | `models/staging/stg_tickets.sql`（同名同职责） |
| `models/intermediate/int_ticket_activity_daily.sql` | 同名中间层，只是维度更少 |
| `models/marts/support_kpi_mart.sql` | 同名 mart；原项目还做了指标转长表（metric_name/metric_value） |
| `tests/kpi_counts_non_negative.sql` 等 | `tests/metric_values_non_negative.sql`、`tests/ratio_metrics_between_0_and_1.sql` |

## 验证基线

`dbt build --profiles-dir .` 应输出：`Done. PASS=31 WARN=0 ERROR=0 SKIP=0`（3 seeds + 5 models + 23 tests）。

## L4 挑战

不看任何资料完成以下动作（这是真懂的检验标准）：

1. 备份后**删除整个 `models/` 目录**和 `tests/` 目录。
2. 凭记忆重建三层结构：staging（用 `source()` 取 seed 数据）→ intermediate（派生 `is_p1`、`sla_breached`）→ marts（聚合 KPI）。
3. 写出至少 3 个 generic tests 和 1 个 singular test。
4. `dbt build --profiles-dir .` 全部跑通。
5. 用 `dbt build --select tag:xxx` 只跑你打过 tag 的一层。

卡住了再回来翻 `../notes/` 里对应的笔记，并记下卡在哪一步——那就是你还没真懂的地方。
