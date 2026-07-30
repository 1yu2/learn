# 02 Models 与物化方式

## 为什么需要物化这个概念

dbt 里 **一个 `.sql` 文件 = 一个 model**，内容就是一条 `select` 语句——不用写 `create table`、不用写 `insert`，dbt 替你包。问题是：这条 select 在数仓里落地成什么？是每次都实时算的视图，还是存下来的物理表？这就是**物化（materialization）**，选错了直接影响查询速度、存储成本和数据新鲜度。

## 一个 model 长什么样

`models/staging/stg_customers.sql`：

```sql
select
    customer_id,
    customer_name,
    plan_tier,
    region
from {{ source('raw', 'raw_customers') }}
```

文件名 `stg_customers` 就是 model 名（也就是数仓里建出来的对象名）。`{{ source(...) }}` / `{{ ref(...) }}` 是 Jinja，编译时替换成真实的表名——下一篇细讲。

## 四种物化方式对比

| 物化 | 数仓里是什么 | 每次 `dbt run` | 查询时 | 适合 |
| --- | --- | --- | --- | --- |
| `view` | 视图（只存 SQL 定义） | 重建视图定义，不算数据 | 实时算，永远最新 | 数据量小、逻辑常变、下游不介意算一遍 |
| `table` | 物理表 | **全删重建**（drop + create） | 直接读，快 | 结果不大、被频繁查询、需要快速响应的报表 |
| `incremental` | 物理表 | 首次全量，之后**只插入/更新新数据** | 直接读，快 | 大数据量的事实表，全量重建太贵 |
| `ephemeral` | **什么都不建** | 不建对象 | 编译时把 SQL 以 CTE 形式内嵌进下游 model | 只想复用一段逻辑、不想污染数仓的中间步骤 |

要点：

- `table` 是全量重建，不是追加。model 有几亿行时全量重建又慢又贵 → 用 `incremental`。
- `incremental` 需要配合 `is_incremental()` 写增量过滤逻辑（stage 2 主题 12 深入），入门阶段知道存在即可。
- `ephemeral` 的 model 在数仓里查不到（没有对象），别对它抱「能 select 看看」的期待。

## 怎么配置物化

**方式一：model 文件顶部的 config 块**（只影响这一个 model）

```sql
{{ config(materialized='table') }}

select ...
```

**方式二：dbt_project.yml 按目录统一配置**（推荐，惯例）

```yaml
models:
  support_kpi:
    staging:
      +materialized: view
    marts:
      +materialized: table
```

优先级：config 块 > dbt_project.yml。所以惯例是「项目文件定默认值，个别 model 特殊时在文件里覆盖」。

`.yml` 属性文件（schema.yml）里也能配，放在 model 的 `config:` 键下，但物化方式通常不放这里——放前两种位置更常见。

## 什么时候用哪种（经验法则）

- **staging 层** → `view`：只是清洗重命名，逻辑轻，让下游实时算。
- **intermediate 层** → `view`（或 ephemeral）：中间加工，没人直接查。
- **marts 层** → `table`：报表/下游系统直接查，要毫秒级响应。
- 事实表大到全量重建不可接受 → `incremental`。

> 对应 omnisupport-copilot：`analytics/dbt_project.yml` 里就是这个约定——staging/intermediate 是 view，marts（含 `support_kpi_mart`）是 table。practice 项目原样照搬了这一约定。

## L4 自检

- [ ] 不看资料说出四种物化在数仓里分别落地成什么、各自适合什么场景。
- [ ] 解释为什么 `table` 物化不适合超大事实表，该换成什么。
- [ ] 在 practice 项目里把 `support_kpi_mart` 改成 view 再改回 table，分别用两种方式（config 块 / dbt_project.yml）各做一次。
- [ ] 用 DuckDB 查 `dev.duckdb`，验证 staging 层是 view、marts 层是 table（`show tables` / `information_schema`）。
- [ ] 新建一个 ephemeral model 并被下游引用，然后在数仓里确认它**没有**对应的对象。
