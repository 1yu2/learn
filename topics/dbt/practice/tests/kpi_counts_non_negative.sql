-- 业务规则：KPI 计数类指标不允许为负数。
-- singular test 返回 0 行 = 通过；返回任何行 = 失败。
select *
from {{ ref('support_kpi_mart') }}
where total_tickets < 0
   or resolved_tickets < 0
   or p1_tickets < 0
   or sla_breach_count < 0
