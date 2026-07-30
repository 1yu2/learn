# 08 项目结构最佳实践：staging / intermediate / marts 三层分层

## 为什么要分层

把所有转换塞进一个 model 的下场：上游表结构一变，一处改处处崩；两个报表要同一个口径，各自复制粘贴，然后慢慢漂移成两个数字。分层就是按「数据的加工程度」把转换拆开，每层只做一件事、只依赖相邻的层，让变化被隔离、复用自然发生。

dbt 官方 best practices 推荐的骨架是三层（目录即层）：

```
models/
├── staging/        # 贴源层：从 source 进来，只做清洗
├── intermediate/   # 中间层：业务加工、拼接、聚合准备
└── marts/          # 集市层：面向业务交付的最终表
```

## 各层职责与命名

### staging（stg_）

- **做什么**：一个 source 表对应一个 stg model。只做轻量清洗：重命名、类型转换、大小写/空值标准化、按需要打平结构。**不做 join、不做业务规则**。
- **为什么**：把「脏的原始世界」封装起来。上游表改名/换类型时只改这一个文件，下游无感。
- **命名**：`stg_<来源>__<实体>` 或简单 `stg_<实体>`，如 `stg_tickets`、`stg_customers`。
- **物化惯例**：`view`（逻辑轻，让下游实时算）。

### intermediate（int_）

- **做什么**：业务加工的主战场。派生业务字段（`is_p1`、`sla_breached`）、join 拼接、为某个 mart 做预聚合。
- **为什么**：复杂逻辑从 mart 里剥离出来后可复用、可测试；多个 mart 共享同一份加工结果。
- **命名**：`int_<动词或业务过程>`，如 `int_ticket_activity_daily`（按天的工单活动加工）。
- **物化惯例**：`view`（没人直接查它）；逻辑被多处复用且很贵时可考虑 ephemeral/table。

### marts（fct_ / dim_）

- **做什么**：面向业务消费者的最终交付。报表、下游系统、AI 工具只准查这一层。
- **命名**：按 Kimball 惯例分**事实表**和**维表**：
  - `fct_<业务过程>`：度量/事件，如 `fct_orders`（行多、数值多）；
  - `dim_<实体>`：描述性宽表，如 `dim_customers`（一行一个实体）。
  - 综合性的指标宽表也常直接叫 `xxx_mart`，如 `support_kpi_mart`。
- **物化惯例**：`table`（被频繁查询，要快）；超大事实表用 `incremental`。

## 分层间的纪律

- 依赖只许**向后**走：staging ← source；intermediate ← staging；marts ← intermediate/staging。不许反向、不许跨层乱引用原始表。
- staging 之后**不许再出现 source()**（见 03 篇）。
- 每个 model 保持「一个文件一个职责」，长得读不下去就拆出一个 int_。

## practice 项目的对照

| 层 | practice 文件 | omnisupport-copilot 对应 |
| --- | --- | --- |
| staging | `stg_tickets.sql`：类型转换、状态小写化、`created_date` 派生 | `analytics/models/staging/stg_tickets.sql`：同样做标准化，还多派生了 `is_open/is_p1/sla_breached`（原项目把部分加工下放到了 staging，是常见的变体） |
| intermediate | `int_ticket_activity_daily.sql`：派生 is_p1/sla_breached，按天+类别聚合 | `analytics/models/intermediate/int_ticket_activity_daily.sql`：同名同职责，维度更多 |
| marts | `support_kpi_mart.sql`：按类别的工单量/解决时长/SLA 达标率 | `analytics/models/marts/support_kpi_mart.sql`：同名，额外把指标转成长表供指标注册表消费 |

各层的物化配置都在 `dbt_project.yml` 里按目录统一声明（staging/intermediate → view，marts → table），并统一打了 tag——两个项目的做法一致。

## L4 自检

- [ ] 不看资料画出三层结构，说清每层的职责、命名前缀、物化惯例。
- [ ] 解释为什么 staging 层不允许 join、为什么 mart 层必须是 table 居多。
- [ ] 给 practice 项目加一个新需求（如按 `plan_tier` 看 SLA 达标率）：说出该动哪一层、不动哪一层，并实现它。
- [ ] 判断题并说明理由：在 mart 里直接 `{{ source('raw', 'raw_tickets') }}` 可以吗？
- [ ] 说出 `fct_` 和 `dim_` 的区别，并把 practice 的三个 staging model 归类为「将来谁会被 join 进 dim、谁会流向 fct」。
