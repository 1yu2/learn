select
    customer_id,
    customer_name,
    plan_tier,
    region
from {{ source('raw', 'raw_customers') }}
