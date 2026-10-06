# Open-source and self-hostable first, hosted version later

IncarnaMind is built first as an open-source app that people run themselves with their own model API keys. A hosted version comes later. Version 1 therefore has no credits, usage quotas, subscription tiers or payments. Django's `credits`, `max_tokens`, `g_type` and `subscription_end_date` are dropped, not ported. Usage limits get designed when the hosted version is real, rather than carried over from a scheme that was never fully used (`g_type` and `subscription_end_date` were never read anywhere).

The hosted version must sync a user's Minds across their devices. Version 1 doesn't need sync, but nothing in its data model may rule sync out.
