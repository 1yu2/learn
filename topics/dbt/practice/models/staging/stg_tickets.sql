with source as (
    select * from {{ source('raw', 'raw_tickets') }}
),

cleaned as (
    select
        ticket_id,
        customer_id,
        subject,
        lower(status) as status,
        lower(nullif(priority, '')) as priority,
        category,
        cast(created_at as timestamp) as created_at,
        cast(sla_due_at as timestamp) as sla_due_at,
        cast(resolved_at as timestamp) as resolved_at,
        cast(first_response_minutes as integer) as first_response_minutes,
        cast(created_at as date) as created_date
    from source
)

select * from cleaned
