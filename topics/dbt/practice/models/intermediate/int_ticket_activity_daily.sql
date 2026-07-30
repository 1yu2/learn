with tickets as (
    select * from {{ ref('stg_tickets') }}
),

enriched as (
    select
        created_date as activity_date,
        category,
        ticket_id,
        case when status in ('open', 'in_progress', 'pending_customer', 'escalated') then true else false end as is_open,
        case when status = 'resolved' then true else false end as is_resolved,
        case when status = 'escalated' then true else false end as is_escalated,
        case when priority = 'p1' then true else false end as is_p1,
        case
            when sla_due_at is null then false
            when coalesce(resolved_at, current_timestamp) > sla_due_at then true
            else false
        end as sla_breached,
        case
            when resolved_at is not null
            then date_diff('hour', created_at, resolved_at)
        end as resolution_hours
    from tickets
)

select
    activity_date,
    category,
    count(*) as ticket_count,
    sum(case when is_open then 1 else 0 end) as open_ticket_count,
    sum(case when is_resolved then 1 else 0 end) as resolved_ticket_count,
    sum(case when is_p1 then 1 else 0 end) as p1_ticket_count,
    sum(case when is_escalated then 1 else 0 end) as escalation_count,
    sum(case when sla_breached then 1 else 0 end) as sla_breach_count,
    avg(resolution_hours) as avg_resolution_hours
from enriched
group by 1, 2
