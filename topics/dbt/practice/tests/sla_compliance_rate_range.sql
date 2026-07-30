-- 业务规则：SLA 达标率必须在 [0, 1] 区间内。
select *
from {{ ref('support_kpi_mart') }}
where sla_compliance_rate < 0 or sla_compliance_rate > 1
