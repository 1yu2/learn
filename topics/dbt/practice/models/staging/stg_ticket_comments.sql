select
    comment_id,
    ticket_id,
    lower(author_type) as author_type,
    cast(commented_at as timestamp) as commented_at
from {{ source('raw', 'raw_ticket_comments') }}
