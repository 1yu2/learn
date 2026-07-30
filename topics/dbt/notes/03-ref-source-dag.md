# 03 ref、source 与 DAG

## 为什么不能在 SQL 里直接写表名

新手直觉：`select * from raw_tickets`。这在 dbt 里是反模式，原因有二：

1. **依赖丢失**：dbt 不知道你的 model 依赖 `raw_tickets`，就没法排执行顺序、没法画血缘图。
2. **环境漂移**：dev 环境和 prod 环境的 schema 不同，硬编码表名就写死了。

所以 dbt 提供两个函数替代硬编码：`ref()` 和 `source()`。

## ref()：引用另一个 model

```sql
select * from {{ ref('stg_tickets') }}
```

含义：「依赖项目里名为 `stg_tickets` 的 model」。编译时被替换成实际的数据库对象（如 `main.stg_tickets`）。**model 之间的引用一律用 ref()**——dbt 据此建立依赖关系。

## source()：引用外部原始数据

原始数据（别的工具抽进数仓的表、或 seed 加载的表）不是 dbt model，不能用 ref，要先**声明**再引用。

**声明**——`models/staging/sources.yml`：

```yaml
version: 2

sources:
  - name: raw                 # source 名（逻辑分组）
    schema: main              # 这些表所在的 schema
    tables:
      - name: raw_tickets
        columns:
          - name: ticket_id
            tests:            # source 上也能直接配测试
              - not_null
              - unique
```

**引用**：

```sql
select * from {{ source('raw', 'raw_tickets') }}
```

即 `{{ source('source名', '表名') }}`。

ref vs source 一句话：**dbt 建的用 ref，dbt 之外来的用 source**。staging 层是两者唯一的交界：staging model 用 source() 取原始数据，之后的所有层只用 ref() 引用其他 model。

> 对应 omnisupport-copilot：`analytics/models/sources.yml` 声明了 `omni_postgres` source（PostgreSQL 里的 ticket_fact、customer_dim 等），`models/staging/stg_tickets.sql` 第一行就是 `{{ source('omni_postgres', 'ticket_fact') }}`。practice 项目里我们声明了 `raw` source 指向 seed 加载的三张表，结构完全对应。

## DAG（有向无环图）

所有 ref/source 关系连起来就是一张 **DAG**：

```
raw_tickets (source) ──► stg_tickets ──► int_ticket_activity_daily ──► support_kpi_mart
raw_customers (source) ──► stg_customers ──► (relationships test)
```

- **有向**：依赖有方向（上游先跑）。
- **无环**：A 依赖 B、B 又依赖 A 是不允许的，dbt 会直接报循环依赖错误。

DAG 是 dbt 一切调度的基础：`dbt build` 按拓扑序执行；`--select stg_tickets+` 里的 `+` 就是沿 DAG 向下游走；文档站里的 lineage 图也是它。

## freshness（源数据新鲜度检查）

原始表是外部工具加载的，可能抽数任务挂了你却不知道。在 source 声明里加：

```yaml
sources:
  - name: omni_postgres
    schema: public
    loaded_at_field: updated_at      # 用哪个字段判断「最后加载时间」
    freshness:
      warn_after: {count: 12, period: hour}
      error_after: {count: 24, period: hour}
    tables:
      - name: ticket_fact
```

然后跑 `dbt source freshness`，超过阈值就 warn/error。注意：**seed 加载的表没有加载时间字段，做不了 freshness**（practice 项目的 raw source 因此没配）；它面向的是生产里的外部源表，omnisupport-copilot 那种连真实 PostgreSQL 的场景才用得上。

## L4 自检

- [ ] 一句话说清 ref 和 source 的分工，以及为什么只有 staging 层该出现 source()。
- [ ] 不看资料给 practice 项目新加一个 seed CSV + source 声明 + staging model，并让它出现在 DAG 里。
- [ ] 故意制造一个循环依赖，观察 dbt 的报错信息，然后修掉。
- [ ] 用 `dbt build --select +support_kpi_mart` 验证你对「上游」方向的理解（应跑全部 seeds 和三层 model）。
- [ ] 解释 freshness 检查解决什么问题、`loaded_at_field` 起什么作用、为什么 seed 表不适用。
