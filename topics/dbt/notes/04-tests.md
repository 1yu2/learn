# 04 Tests（数据测试）

## 为什么数据也要测试

代码有单元测试，数据转换逻辑同样会出错：join 出重复行、上游出现意料之外的枚举值、主键悄悄出现 NULL。没有测试，这些错误会无声无息流进报表，直到业务方发现数字对不上。dbt 把「给数据写断言」变成一等公民：**测试是 SQL 查询，查出 0 行 = 通过**。

dbt 的测试分两类：generic（通用的、声明式）和 singular（单独的、手写 SQL）。

## Generic tests：yml 里声明

dbt 内置四个最常用的 generic tests，写在 model 旁的 `schema.yml`（任意 `*.yml` 均可）里：

```yaml
version: 2

models:
  - name: stg_tickets
    columns:
      - name: ticket_id
        tests:
          - not_null                          # 不允许 NULL
          - unique                            # 不允许重复
      - name: status
        tests:
          - accepted_values:                  # 值必须在枚举内
              arguments:
                values: ["open", "in_progress", "pending_customer", "resolved", "escalated"]
      - name: customer_id
        tests:
          - relationships:                    # 外键：每个 customer_id 都存在于客户表
              arguments:
                to: ref('stg_customers')
                field: customer_id
```

| 测试 | 断言 | 典型用途 |
| --- | --- | --- |
| `not_null` | 列无 NULL | 主键、关键维度 |
| `unique` | 列无重复 | 主键、粒度键 |
| `accepted_values` | 值在枚举列表内 | 状态、优先级 |
| `relationships` | 每行的值都存在于目标表的指定列 | 外键完整性 |

注意当前版本语法：带参数的测试要把参数放在 `arguments:` 键下。

测试也可以直接挂在 **source** 的列上（见 `models/staging/sources.yml`），在数据进门的第一刻就校验——omnisupport-copilot 的 `analytics/models/sources.yml` 就是这么做的。

## Singular tests：tests/ 目录下的 SQL 文件

内置四种覆盖不了的业务规则，就手写一个 SQL 放 `tests/` 目录：**查出问题行，返回 0 行即通过**。

`tests/kpi_counts_non_negative.sql`（practice 项目）：

```sql
-- 业务规则：KPI 计数类指标不允许为负数
select *
from {{ ref('support_kpi_mart') }}
where total_tickets < 0
   or resolved_tickets < 0
   or p1_tickets < 0
   or sla_breach_count < 0
```

写 singular test 的心法是**反着写**：不是「查出正确数据」，而是「查出违反规则的数据」，查不到才算对。

> 对应 omnisupport-copilot：`analytics/tests/metric_values_non_negative.sql` 和 `ratio_metrics_between_0_and_1.sql` 与此完全相同——practice 项目的两个 singular test 就是照它们简化来的。

## 运行

```bash
dbt test                                    # 全部测试
dbt test --select stg_tickets               # 某个 model 上的测试
dbt test --select test_type:generic         # 只跑 generic
dbt test --select test_type:singular        # 只跑 singular
dbt build                                   # 推荐：model 建完立即跑它的测试，失败阻断下游
```

测试失败时，dbt 会把出错行数报出来，编译后的测试 SQL 在 `target/compiled/` 下，可以直接拿去数仓里跑，看看到底是哪些行违规。

## L4 自检

- [ ] 不看资料写出四个内置 generic tests 的名字和各自断言语义。
- [ ] 在 practice 项目里故意把 seed 数据改坏（比如造一个重复 ticket_id），看哪个测试怎么失败，再改回来。
- [ ] 手写一个新的 singular test（如「resolved 的工单 resolved_at 不为空」），先让它通过，再改数据让它失败。
- [ ] 解释 singular test「返回 0 行 = 通过」的设计，以及为什么要「反着写」。
- [ ] 说出 generic 和 singular 的适用边界：什么情况必须从 yml 声明升级为手写 SQL。
