{{ config(tags=['kpi']) }}

with daily as (
    select * from {{ ref('int_ticket_activity_daily') }}
)

select
    category,
    sum(ticket_count) as total_tickets,
    sum(open_ticket_count) as open_tickets,
    sum(resolved_ticket_count) as resolved_tickets,
    sum(p1_ticket_count) as p1_tickets,
    sum(escalation_count) as escalation_count,
    sum(sla_breach_count) as sla_breach_count,
    sum(ticket_count * avg_resolution_hours) / nullif(sum(resolved_ticket_count), 0) as avg_resolution_hours,
    1.0 - sum(sla_breach_count) * 1.0 / nullif(sum(ticket_count), 0) as sla_compliance_rate
from daily
group by 1
