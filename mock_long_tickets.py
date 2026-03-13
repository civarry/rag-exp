"""
Long-form Jira tickets for testing chunking strategies on realistic-length documents.

Each ticket is 5-15KB with:
- Detailed descriptions
- Long comment threads with debugging back-and-forth
- Stack traces, log excerpts, config snippets
- Multiple contributors with different writing styles
"""

LONG_TICKETS = [
    {
        "key": "LONG-101",
        "summary": "Production database failover caused 23-minute outage during peak hours",
        "type": "Incident",
        "priority": "Critical",
        "description": """
Incident Report: Production Database Failover Failure
Date: 2024-01-15 14:32 UTC
Duration: 23 minutes
Severity: SEV-1
Affected Services: All customer-facing APIs, admin dashboard, webhook processing

## Timeline

14:32 - Primary PostgreSQL node (db-primary-us-east-1a) reported disk I/O errors. CloudWatch alarm triggered.
14:33 - Automated failover initiated by RDS Multi-AZ. Secondary node (db-secondary-us-east-1b) promoted to primary.
14:34 - Connection pool errors across all application pods. PgBouncer reported "server login failed: FATAL password authentication failed for user app_service".
14:35 - On-call engineer @mike.chen paged. Acknowledged at 14:37.
14:38 - Investigation revealed the failover node had a different pg_hba.conf configuration. The secondary was provisioned 6 months ago during a DR drill but the pg_hba.conf was never synced with the primary's updated rules that were added in November for the new microservices.
14:40 - Attempted to manually update pg_hba.conf on the new primary. Required SSH access which was blocked by the security group.
14:42 - @sarah.ops escalated to infrastructure team. Security group modification initiated.
14:45 - Security group updated. SSH access established.
14:47 - pg_hba.conf updated to match primary configuration. PostgreSQL reload initiated.
14:50 - Application pods still failing. Investigation showed PgBouncer had cached the old connection and was not retrying.
14:52 - PgBouncer restart initiated across all 12 pods. Rolling restart to minimize additional downtime.
14:55 - First pods reconnected. Partial service restoration confirmed via health checks.
14:55 - Remaining pods reconnected. Full service restoration confirmed.

## Impact

- 23 minutes of complete API unavailability
- ~4,200 failed API requests (tracked via API gateway 503 responses)
- ~890 webhook deliveries delayed (queued in RabbitMQ, delivered after recovery)
- 12 customer-reported incidents via support portal
- Estimated revenue impact: $18,400 based on average transaction rate during affected period

## Root Cause

The failover PostgreSQL instance had a stale pg_hba.conf that did not include authentication rules for three microservices deployed after the DR drill in July 2024. When failover occurred, these services could not authenticate, causing cascading connection failures across the entire application stack due to shared PgBouncer pools.

## Contributing Factors

1. No automated sync mechanism for pg_hba.conf between primary and secondary
2. DR drill checklist did not include verification of configuration parity
3. PgBouncer connection caching prevented automatic recovery even after the database was accessible
4. SSH access to RDS instances required security group modification, adding 5 minutes to response time
        """.strip(),
        "comments": [
            # Comment 1 - Incident commander
            """@all Post-incident review scheduled for Thursday 2024-01-18 at 10:00 UTC.

Action items from initial triage:
1. [P0] Implement automated pg_hba.conf sync between primary and failover - @sarah.ops
2. [P0] Add PgBouncer health check that detects stale connections and auto-restarts - @mike.chen
3. [P1] Pre-authorize SSH access to RDS instances for on-call engineers - @infra-team
4. [P1] Add pg_hba.conf parity check to DR drill checklist - @sarah.ops
5. [P2] Investigate connection pool library alternatives that handle failover gracefully - @backend-team

Customer communication was handled by @support-lead. All affected customers received incident notification within 15 minutes of detection.""",

            # Comment 2 - Backend engineer debugging
            """Did some deeper investigation into why PgBouncer didn't recover automatically.

The issue is in our PgBouncer config:
```
server_check_delay = 30
server_check_query = select 1
server_login_retry = 0
server_connect_timeout = 15
```

`server_login_retry = 0` means PgBouncer will NOT retry a failed login. Once the auth failure happens, that server connection is marked dead and PgBouncer needs a full restart to try again. This is documented in the PgBouncer FAQ but it's a non-obvious default.

Fix: Set `server_login_retry = 3` and add `server_reconnect_timeout = 5` so PgBouncer retries auth failures automatically. With this config, the 23-minute outage would have been ~3 minutes (time for failover + one retry cycle).

PR #1247 ready for review.""",

            # Comment 3 - Infrastructure engineer
            """Regarding the pg_hba.conf sync - I've prototyped a solution using AWS Systems Manager Parameter Store.

Approach:
1. pg_hba.conf is stored as an SSM parameter (versioned)
2. Both primary and secondary run a cron job every 5 minutes that pulls from SSM and diffs against local
3. If diff detected, update local and `pg_ctl reload`
4. CloudWatch alarm if sync fails

Tested in staging - works reliably. One concern: the cron approach means up to 5 minutes of config drift. For pg_hba.conf this is acceptable since auth rules don't change frequently. But we should NOT use this pattern for postgresql.conf tuning parameters.

Alternative considered: AWS RDS parameter groups handle most PostgreSQL configs but NOT pg_hba.conf custom rules. That's an RDS limitation we've hit before.

PR #1251 - SSM-based pg_hba.conf sync""",

            # Comment 4 - On-call engineer post-mortem notes
            """Post-mortem notes from the review meeting:

Root cause confirmed as pg_hba.conf drift. But we also identified a secondary issue that made detection harder:

Our monitoring only checks "can the APPLICATION connect to the database" - it doesn't check "can ALL services connect." During the outage, the health check service (which uses a legacy connection string with a different user) was still connecting fine. So our primary health check was GREEN while three microservices were completely down.

Action item added:
6. [P0] Health checks must verify connectivity for ALL service accounts, not just the health check user - @mike.chen

Also worth noting: the PgBouncer logs had the exact error message but our log aggregation pipeline has a 3-minute delay for PostgreSQL/PgBouncer logs (they go through a different Fluentd pipeline than application logs). If we'd seen the auth error immediately, we could have diagnosed in under 2 minutes instead of 8.

7. [P1] Reduce PgBouncer log aggregation delay to <30 seconds - @platform-team""",

            # Comment 5 - SRE follow-up
            """All P0 action items completed as of 2024-01-25:

1. ✅ pg_hba.conf SSM sync deployed (PR #1251 merged)
2. ✅ PgBouncer server_login_retry fix deployed (PR #1247 merged)
3. ✅ Health checks updated to verify all service accounts (PR #1259 merged)

Validation: Ran a controlled failover test in staging on 2024-01-24. Results:
- Failover time: 45 seconds (RDS automatic)
- PgBouncer reconnection: 12 seconds (with retry config)
- All services recovered: 62 seconds total
- Zero failed health checks after PgBouncer reconnection

This is down from the 23-minute outage to ~1 minute. Scheduling production validation for next maintenance window (2024-02-01).

P1 items in progress:
- SSH pre-authorization: security review pending
- DR checklist update: merged into runbook
- PgBouncer log pipeline: Fluentd config updated, testing"""
        ],
    },
    {
        "key": "LONG-201",
        "summary": "Memory leak in order processing service causing OOM kills every 48 hours",
        "type": "Bug",
        "priority": "High",
        "description": """
## Problem

The order-processing-service pods are being OOM-killed by Kubernetes approximately every 48 hours. The service starts with ~256MB RSS and grows steadily at approximately 2.1MB/hour until hitting the 512MB memory limit.

Current workaround: Kubernetes restarts the pod automatically, but during the restart window (~30 seconds), orders are queued in RabbitMQ and processed after restart. This causes intermittent 30-60 second delays in order confirmation emails.

## Environment

- Service: order-processing-service v4.2.1
- Language: Python 3.11
- Framework: FastAPI + Celery
- Memory limit: 512MB (Kubernetes resource limit)
- Pods: 3 replicas
- Queue: RabbitMQ 3.12

## Evidence

Memory graph from Grafana (last 7 days) shows a clear sawtooth pattern:
- Pod 1: OOM at 01/10 03:22, 01/12 04:15, 01/14 02:58
- Pod 2: OOM at 01/10 15:44, 01/12 16:30, 01/14 17:12
- Pod 3: OOM at 01/11 08:33, 01/13 09:20, 01/15 10:05

Each restart resets to ~256MB and the growth begins again immediately.

## Steps to Reproduce

1. Deploy order-processing-service v4.2.1
2. Run load test: 50 orders/minute sustained
3. Monitor RSS via `kubectl top pods`
4. After ~24 hours, RSS will be ~300MB
5. After ~48 hours, OOM kill at 512MB

## Initial Investigation

Ran `tracemalloc` in staging with production-like load for 6 hours. Top allocations:

```
/app/services/order_processor.py:142: size=45.2 MiB, count=892341, average=53 B
/app/services/inventory_client.py:89: size=23.1 MiB, count=234521, average=103 B
/usr/local/lib/python3.11/site-packages/celery/worker/request.py:231: size=12.4 MiB
/app/utils/cache.py:34: size=8.7 MiB, count=45123, average=202 B
```

The order_processor.py:142 allocation is suspicious — it's in the `_build_order_context` method which creates a dict of order details for downstream processing. But the dict should be garbage collected after each order is processed.
        """.strip(),
        "comments": [
            # Comment 1 - Developer investigating
            """I spent 3 hours profiling this today. Found the leak.

In `order_processor.py`, the `OrderProcessor` class has a class-level dict `_processing_cache` that was intended as a per-request cache but is actually shared across all instances:

```python
class OrderProcessor:
    _processing_cache = {}  # THIS IS THE LEAK - class variable, never cleared

    def process_order(self, order):
        context = self._build_order_context(order)
        self._processing_cache[order.id] = context  # grows forever
        # ... process order ...
        # BUG: _processing_cache is never cleaned up
```

Every order processed adds ~2KB to `_processing_cache`. At 50 orders/minute, that's 100KB/min = 6MB/hour = 144MB/day. Matches the ~2.1MB/hour growth rate when accounting for the context objects having references to other cached objects.

The fix is straightforward - either:
A) Use an instance variable instead of class variable + clear after processing
B) Use an LRU cache with a max size
C) Just delete the entry after processing completes

I'm going with option C as the simplest fix, plus adding a safety net LRU with maxsize=1000:

```python
from functools import lru_cache

class OrderProcessor:
    def __init__(self):
        self._processing_cache = {}  # instance variable now

    def process_order(self, order):
        context = self._build_order_context(order)
        self._processing_cache[order.id] = context
        try:
            result = self._execute_order(context)
        finally:
            self._processing_cache.pop(order.id, None)  # always clean up
        return result
```

PR #892 ready for review.""",

            # Comment 2 - Code reviewer
            """Good catch @alex. Reviewed PR #892, couple of things:

1. The fix looks correct but we also have the same pattern in `inventory_client.py:89` — the `_response_cache` dict there also grows unbounded. It's the second biggest allocator in the tracemalloc output. Can you fix that too?

2. I'd suggest we also add a prometheus metric for `_processing_cache` size so we can alert if it grows beyond expected bounds in the future:
```python
from prometheus_client import Gauge
cache_size = Gauge('order_processor_cache_size', 'Current size of processing cache')
```

3. Minor: the `finally` block is good but we should also handle the case where `order.id` is None (happens with draft orders). Add a guard:
```python
if order.id is not None:
    self._processing_cache.pop(order.id, None)
```

Otherwise LGTM. Fix the inventory client leak too and ship it.""",

            # Comment 3 - Developer responding
            """Updated PR #892:

1. ✅ Fixed `inventory_client.py` — same pattern, `_response_cache` class variable changed to instance variable with cleanup in finally block. This was leaking ~1MB/hour on its own.

2. ✅ Added Prometheus gauge for both caches. Dashboard updated: grafana.internal/d/order-processing-memory

3. ✅ Added None guard for draft orders.

Also found a THIRD leak while fixing the inventory client — `utils/cache.py:34` has a homegrown TTL cache that uses `time.time()` for expiry checks but the cleanup thread was using `time.monotonic()`. The clock skew meant entries were effectively never expired. Fixed by using `time.monotonic()` consistently.

Load test results after fix (6 hours, 50 orders/min):
- Before: 256MB → 268MB (+12MB, 2MB/hour)
- After: 256MB → 257.2MB (+1.2MB, 0.2MB/hour)

The remaining 0.2MB/hour growth is likely normal Python heap fragmentation. At that rate, the pod would last ~53 days before hitting 512MB. We could increase the memory limit to 768MB for extra safety margin.

Ready for re-review.""",

            # Comment 4 - Tech lead sign-off
            """Reviewed the updated PR. All three leaks addressed, load test numbers look good.

Regarding the 0.2MB/hour residual — that's expected. Python's memory allocator (pymalloc) doesn't always return memory to the OS even after deallocation. The 53-day estimate is conservative; in practice the fragmentation stabilizes.

I'd keep the 512MB limit for now. If we see it creeping up again we can investigate further, but 53 days between restarts is fine for a service that gets redeployed weekly anyway.

Approving PR #892. Let's deploy to staging today and monitor for 48 hours before promoting to prod.

@devops please schedule the staging deployment.""",

            # Comment 5 - DevOps deployment confirmation
            """Deployed order-processing-service v4.2.2 (PR #892) to staging at 2024-01-16 09:00 UTC.

48-hour monitoring results:
- Pod 1: 256MB → 258.4MB (stable)
- Pod 2: 256MB → 257.8MB (stable)
- Pod 3: 256MB → 258.1MB (stable)

No OOM kills. Prometheus cache_size gauge holding steady at 0-15 entries (peak during high traffic).

Promoting to production. Deployment scheduled for 2024-01-18 maintenance window.

UPDATE 2024-01-19: Production deployment successful. 72 hours monitoring shows identical stable memory pattern. Closing this ticket."""
        ],
    },
    {
        "key": "LONG-301",
        "summary": "Implement rate limiting for public API with tiered customer plans",
        "type": "Feature",
        "priority": "High",
        "description": """
## Background

Our public API currently has no rate limiting. We've had three incidents in the past month where a single customer's automated scripts overwhelmed the API, causing degraded performance for all customers. We need to implement tiered rate limiting based on customer plan type.

## Requirements

### Rate Limit Tiers

| Plan | Requests/minute | Requests/hour | Burst (requests/second) |
|------|----------------|---------------|------------------------|
| Free | 60 | 1,000 | 5 |
| Starter | 300 | 10,000 | 20 |
| Business | 1,000 | 50,000 | 50 |
| Enterprise | 5,000 | 200,000 | 200 |

### Implementation Requirements

1. Rate limiting must be applied at the API gateway level (Kong)
2. Limits are per API key, not per IP address
3. Must return standard rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
4. When rate limited, return HTTP 429 with a Retry-After header
5. Rate limit counters must survive gateway restarts (use Redis)
6. Dashboard showing rate limit hits per customer (Grafana)
7. Alerting when a customer hits >80% of their limit (PagerDuty)

### Architecture Decision

After evaluating options:

Option A: Kong rate-limiting plugin (built-in)
- Pros: Simple, well-tested, native Kong integration
- Cons: Limited customization, no tiered support without enterprise license

Option B: Custom Kong plugin + Redis
- Pros: Full control, tiered support, custom headers
- Cons: More code to maintain, need to handle Redis failures

Option C: Application-level rate limiting (middleware)
- Pros: No Kong dependency, easier to test
- Cons: Doesn't protect against DDoS at the gateway level, slower

**Decision: Option B** — Custom Kong plugin with Redis backend. This gives us tiered support and the protection of gateway-level enforcement.

### Redis Schema

Using the sliding window log algorithm for accurate rate limiting:

```
Key format: ratelimit:{api_key}:{window_type}:{window_id}
Example: ratelimit:ak_abc123:minute:2024-01-15T14:32
TTL: window_size * 2 (for cleanup)

MULTI
  ZADD ratelimit:ak_abc123:minute:current_window <timestamp> <request_id>
  ZREMRANGEBYSCORE ratelimit:ak_abc123:minute:current_window 0 <window_start>
  ZCARD ratelimit:ak_abc123:minute:current_window
EXEC
```

### Graceful Degradation

If Redis is unavailable:
1. First 30 seconds: Allow all requests (open circuit)
2. After 30 seconds: Fall back to in-memory rate limiting with conservative limits (50% of plan limits)
3. Alert on-call immediately
4. Log all requests that would have been rate-limited for post-incident analysis
        """.strip(),
        "comments": [
            # Comment 1 - Implementation progress
            """Sprint 1 progress update:

Completed:
- Kong plugin skeleton created (`kong-plugins/rate-limiter-tiered/`)
- Redis connection pool with circuit breaker (using lua-resty-redis with 100ms timeout)
- Sliding window implementation — tested with 10K requests, accuracy within 0.5% of expected limits
- Rate limit headers (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset) working correctly

In progress:
- Tier lookup from API key → plan mapping (currently hardcoded, need to integrate with billing service)
- Grafana dashboard for rate limit metrics
- 429 response body formatting

Blocked:
- Need billing service API endpoint to look up plan type by API key. @billing-team can you add GET /internal/api-keys/{key}/plan? We need: plan_name, rate_limits object, and is_active boolean.

Test results from staging load test:
```
Free tier (60 req/min): 60 allowed, 61st rejected ✅
Starter tier (300 req/min): 300 allowed, 301st rejected ✅
Burst limit (5 req/sec free): 5 allowed in 1 second, 6th queued then allowed next second ✅
Redis failover: Circuit breaker opened after 3 failures, in-memory fallback activated ✅
```""",

            # Comment 2 - Billing team response
            """@backend-team The billing service endpoint is ready:

```
GET /internal/api-keys/{key}/plan
Response:
{
  "api_key": "ak_abc123",
  "customer_id": "cust_456",
  "plan": "business",
  "rate_limits": {
    "requests_per_minute": 1000,
    "requests_per_hour": 50000,
    "burst_per_second": 50
  },
  "is_active": true,
  "expires_at": "2024-12-31T23:59:59Z"
}
```

Note: Response is cached on our side with 5-minute TTL. If a customer upgrades their plan, there's up to 5 minutes before the new limits take effect. Is that acceptable?

Also — we have 23 enterprise customers with custom rate limits that don't match the standard tiers. Their limits are stored in a custom_rate_limits field. Make sure your plugin checks that field first before falling back to the plan defaults.""",

            # Comment 3 - Security review
            """Security review of the rate limiter implementation:

Issues found:
1. **[HIGH]** The API key is stored in plain text in Redis keys. If Redis is compromised, all API keys are exposed. Fix: Use a hash of the API key (SHA-256) as the Redis key instead of the raw key.

2. **[MEDIUM]** The billing service endpoint `/internal/api-keys/{key}/plan` is not authenticated. Any pod in the cluster can query any API key's plan details. Add internal service authentication (mTLS or shared secret).

3. **[LOW]** Rate limit headers expose the exact limit to potential attackers, making it easier to optimize their abuse pattern. This is standard practice and acceptable, but consider not including X-RateLimit-Remaining for the Free tier.

4. **[INFO]** The circuit breaker opens after 3 Redis failures. Consider what happens during a Redis cluster failover — Sentinel promotion takes 5-15 seconds, during which you'd get 3 failures and open the circuit. The 30-second open state might be too long. Suggest: reduce to 10 seconds, or detect Sentinel failover events specifically.

Blocking on issue #1. The others can be addressed in follow-up PRs.""",

            # Comment 4 - Developer addressing security
            """All security issues addressed:

1. ✅ **[HIGH]** API key hashing implemented. Redis keys now use `SHA-256(api_key)` prefix. Example: `ratelimit:sha256_a1b2c3...:minute:2024-01-15T14:32`. Raw API key never stored in Redis.

2. ✅ **[MEDIUM]** Added mTLS between Kong and billing service. Using the existing service mesh certificates.

3. ✅ **[LOW]** Kept headers for all tiers after discussion with product — transparency is preferred. Added to API docs.

4. ✅ **[INFO]** Circuit breaker tuned: open threshold reduced to 10 seconds, added Sentinel SUBSCRIBE for failover detection. During Sentinel failover, the circuit breaker enters "half-open" state and retries with the new primary.

Additional change: Added request logging for rate-limited requests. Logs include: hashed API key, endpoint, timestamp, current count, limit. These go to a separate Elasticsearch index for abuse analysis.

PR #1312 updated and ready for re-review.""",

            # Comment 5 - Load test results
            """Final load test results before production deployment:

Test environment: 3 Kong nodes, 3 Redis Sentinel nodes, production-equivalent traffic patterns

**Accuracy Test (1 hour, mixed tiers):**
- Free tier: 60/min limit, measured 60.0/min average (100% accurate)
- Starter tier: 300/min limit, measured 299.8/min average (99.9% accurate)
- Business tier: 1000/min limit, measured 998.2/min average (99.8% accurate)
- Enterprise tier: 5000/min limit, measured 4991.5/min average (99.8% accurate)

**Latency Impact:**
- P50 latency increase: 0.3ms (Redis lookup)
- P99 latency increase: 1.2ms (Redis + billing service cache miss)
- P99.9 latency increase: 4.8ms (circuit breaker state check + fallback)

**Failover Test:**
- Redis primary killed: 2.1 seconds to detect and open circuit breaker
- In-memory fallback activated: 0 requests dropped during transition
- Redis Sentinel promotion: 8.3 seconds, circuit breaker half-open detected new primary
- Full recovery: 11.2 seconds total, 0 requests incorrectly rate-limited

**Abuse Simulation:**
- Simulated 10x limit burst from single API key: correctly limited after first window
- Simulated distributed attack (100 IPs, 1 API key): correctly limited per API key
- Simulated API key rotation mid-window: new key gets fresh counters (expected behavior)

Production deployment scheduled for 2024-02-01. Rollout plan: 10% traffic → 50% → 100% over 3 hours.""",

            # Comment 6 - Post-deployment
            """Deployed to production 2024-02-01. Rollout completed by 15:00 UTC.

First week metrics:
- 847 rate limit hits across 23 customers
- Top offender: customer "data-sync-corp" (Free tier) hitting limit 200+ times/day — reached out to suggest they upgrade to Starter
- 0 false positives reported
- Redis latency stable at 0.2-0.4ms P99
- No circuit breaker activations

One issue discovered: Customer "big-retail-inc" (Enterprise tier) has a custom limit of 10,000 req/min but their actual usage peaks at 8,500/min. The 80% alert threshold (8,000 req/min) is triggering false PagerDuty alerts. Adjusted their alert threshold to 90%.

Overall: rate limiting is working as designed. Moving to monitoring-only mode for the next 2 weeks before closing this ticket."""
        ],
    },
    {
        "key": "LONG-401",
        "summary": "Search indexing pipeline drops documents silently when Elasticsearch bulk API returns partial failures",
        "type": "Bug",
        "priority": "Critical",
        "description": """
## Problem

We discovered that approximately 3.2% of product catalog updates are being silently dropped during Elasticsearch indexing. Products are updated in our PostgreSQL database but never appear in search results. Customer support has received 47 tickets in the past 2 weeks from merchants reporting "my product changes aren't showing up in search."

## Impact

- 3.2% of product updates lost (~1,600 products out of 50,000 daily updates)
- Merchants unable to find their own updated products
- Stale prices in search results causing customer complaints
- Revenue impact: estimated $45K/month in lost sales from stale/missing products

## Root Cause (preliminary)

The search indexing pipeline uses Elasticsearch's `_bulk` API to batch index product updates. The bulk API returns a response with per-document status, but **our code only checks the top-level HTTP status code (200) and ignores individual document errors**.

The bulk response looks like:
```json
{
  "took": 30,
  "errors": true,
  "items": [
    {"index": {"_id": "prod_1", "status": 200, "result": "updated"}},
    {"index": {"_id": "prod_2", "status": 429, "error": {"type": "es_rejected_execution_exception", "reason": "rejected execution of coordinating operation"}}},
    {"index": {"_id": "prod_3", "status": 200, "result": "updated"}}
  ]
}
```

Our code:
```python
response = es_client.bulk(body=bulk_body)
if response.status_code == 200:
    logger.info(f"Bulk indexed {len(docs)} documents")
    # BUG: Never checks response["errors"] or individual item statuses
```

The 429 errors happen when ES is under load — the thread pool queue is full and it rejects some operations. This is intermittent and load-dependent, which is why it wasn't caught in testing (low-volume staging environment).

## Affected Code

- `search_indexer/bulk_indexer.py` — main bulk indexing logic
- `search_indexer/retry_handler.py` — retry logic (only retries on HTTP-level failures, not item-level)
- `search_indexer/monitoring.py` — success metrics (counts requests, not documents)
        """.strip(),
        "comments": [
            # Comment 1 - Developer investigation
            """Confirmed the root cause. Dug deeper into the failure patterns:

Analysis of last 30 days of bulk API responses (parsed from raw ES logs):

```
Total bulk requests: 142,847
Requests with errors=true: 4,573 (3.2%)
Individual document failures breakdown:
  - 429 es_rejected_execution: 78% of failures
  - 409 version_conflict: 15% of failures
  - 400 mapper_parsing_exception: 5% of failures
  - 413 circuit_breaking_exception: 2% of failures
```

The 429 errors are the main problem. They correlate with peak indexing times (2PM-4PM UTC when the bulk product feed runs). During these windows, the ES cluster's write thread pool (size: 8, queue: 200) fills up.

The 409 version conflicts are expected (concurrent updates to the same product) and currently handled correctly — the newer version wins.

The 400 mapper parsing exceptions are a separate bug — 5% of failures are from products with malformed rich text in descriptions that contain invalid UTF-8 sequences. Need a separate ticket for that.

The 413 circuit breaker exceptions happen when a single bulk request is too large (>100MB). Our current batch size is 500 documents but some product descriptions are huge (up to 500KB each). Need to implement size-based batching, not just count-based.""",

            # Comment 2 - Proposed fix
            """Here's my proposed fix for bulk_indexer.py. Three changes:

**1. Parse individual item errors and retry failed documents:**
```python
def bulk_index(self, documents):
    response = self.es_client.bulk(body=self._build_bulk_body(documents))

    if not response.get("errors"):
        self.metrics.record_success(len(documents))
        return

    # Separate successes from failures
    failed_docs = []
    permanent_failures = []

    for item, doc in zip(response["items"], documents):
        action_result = item.get("index") or item.get("update")
        status = action_result.get("status", 200)

        if status in (429, 503):  # Retryable
            failed_docs.append(doc)
        elif status >= 400:  # Permanent failure
            permanent_failures.append((doc, action_result.get("error")))
        # else: success

    self.metrics.record_partial(
        success=len(documents) - len(failed_docs) - len(permanent_failures),
        retryable=len(failed_docs),
        permanent=len(permanent_failures)
    )

    if failed_docs:
        self._retry_with_backoff(failed_docs, max_retries=3)

    if permanent_failures:
        self._send_to_dead_letter_queue(permanent_failures)
```

**2. Size-based batching:**
Instead of fixed 500-document batches, accumulate until batch size hits 50MB:
```python
def _batch_by_size(self, documents, max_bytes=50_000_000):
    batch = []
    batch_size = 0
    for doc in documents:
        doc_size = len(json.dumps(doc).encode('utf-8'))
        if batch_size + doc_size > max_bytes and batch:
            yield batch
            batch = []
            batch_size = 0
        batch.append(doc)
        batch_size += doc_size
    if batch:
        yield batch
```

**3. Exponential backoff for retries:**
```python
def _retry_with_backoff(self, docs, max_retries=3):
    for attempt in range(max_retries):
        wait = min(2 ** attempt * 0.5, 10)  # 0.5s, 1s, 2s
        time.sleep(wait)
        response = self.es_client.bulk(body=self._build_bulk_body(docs))
        if not response.get("errors"):
            self.metrics.record_retry_success(len(docs), attempt + 1)
            return
        # Filter out successes, keep only still-failing docs
        docs = self._extract_failed_docs(response, docs)
    # All retries exhausted
    self._send_to_dead_letter_queue([(d, "max retries exceeded") for d in docs])
```

PR #2041 ready for review.""",

            # Comment 3 - Code review
            """Solid fix. Two issues:

1. The dead letter queue — where does it go? We need a way to replay these. Suggest: write to a `search_indexing_failures` PostgreSQL table with the document JSON, error details, and a `retried_at` column. Then a scheduled job can retry permanent failures every hour.

2. Thread safety — `_retry_with_backoff` uses `time.sleep()` which blocks the Celery worker. Since we run 8 concurrent Celery workers and retries can stack up, we could exhaust all workers on retries while new indexing tasks queue up. Use Celery's `self.retry()` mechanism instead, which properly handles concurrency:

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def bulk_index_task(self, documents):
    try:
        indexer.bulk_index(documents)
    except RetryableIndexError as e:
        self.retry(exc=e, countdown=2 ** self.request.retries)
```

Also — have you considered what happens to the metrics during the retry window? The current code records partial success immediately, but if the retry later succeeds, we'd double-count the successful documents. Need to track "pending retry" as a separate state.""",

            # Comment 4 - Final implementation
            """Updated PR #2041 with review feedback:

1. ✅ Dead letter queue → PostgreSQL `search_indexing_failures` table:
```sql
CREATE TABLE search_indexing_failures (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    document_json JSONB NOT NULL,
    error_type VARCHAR(100),
    error_detail TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    retried_at TIMESTAMP,
    resolved_at TIMESTAMP,
    retry_count INTEGER DEFAULT 0
);
CREATE INDEX idx_failures_unresolved ON search_indexing_failures(created_at) WHERE resolved_at IS NULL;
```

Hourly retry job implemented as a Celery beat task. Retries unresolved failures up to 10 times with exponential backoff.

2. ✅ Switched to Celery's native retry mechanism. Removed `time.sleep()` calls.

3. ✅ Fixed metrics double-counting. New states: `indexed`, `pending_retry`, `retry_success`, `permanent_failure`. Grafana dashboard updated with new panels.

Load test results with the fix (simulating 5% ES rejection rate):
- Before fix: 5% documents silently lost
- After fix: 0% documents lost. 4.8% retried, 4.7% retry success, 0.1% sent to DLQ (mapper errors)
- DLQ replay: 0.08% resolved after data cleanup, 0.02% genuine bad data (logged and alerted)

Deployment plan: This should go out ASAP given the $45K/month revenue impact. Requesting expedited review.""",

            # Comment 5 - Post-deployment
            """Deployed to production 2024-02-05.

First 72 hours monitoring:
- Bulk requests with partial failures: 3.1% (unchanged — these are ES-level rejections)
- Documents retried: 4,892
- Retry success rate: 98.2%
- Documents sent to DLQ: 89 (all mapper_parsing_exception from bad UTF-8)
- Documents silently lost: 0 ✅

The DLQ retry job recovered 67 of the 89 failures after the nightly data cleanup job fixed the UTF-8 issues in PostgreSQL. The remaining 22 are genuinely corrupt product descriptions that need manual merchant intervention.

Customer support confirmed: zero new "product not appearing in search" tickets since deployment. The 47 existing tickets have been resolved — all affected products are now properly indexed.

Revenue recovery estimated at $45K/month. Filing separate ticket LONG-402 for the UTF-8 sanitization issue."""
        ],
    },
    {
        "key": "LONG-501",
        "summary": "Migrate authentication from session cookies to OAuth 2.0 + PKCE for mobile app support",
        "type": "Feature",
        "priority": "High",
        "description": """
## Background

Our web application currently uses server-side session cookies for authentication. We're launching mobile apps (iOS and Android) in Q2, and cookies don't work well for native mobile clients. We need to migrate to OAuth 2.0 with PKCE (Proof Key for Code Exchange) to support both web and mobile clients securely.

## Current State

- Authentication: Express.js + express-session + Redis session store
- Session cookie: `connect.sid`, httpOnly, secure, sameSite=strict
- Session TTL: 24 hours
- Users: ~45,000 active monthly
- Login methods: email/password, Google OAuth (social login)

## Target Architecture

### OAuth 2.0 Authorization Server

Implement an OAuth 2.0 authorization server (using `node-oidc-provider` library) that issues:
- Access tokens (JWT, 15-minute expiry)
- Refresh tokens (opaque, 30-day expiry, rotation on use)
- ID tokens (OIDC-compliant, for user profile info)

### Token Storage

| Client Type | Access Token | Refresh Token |
|-------------|-------------|---------------|
| Web (SPA) | Memory only (no localStorage) | HttpOnly cookie (rotating) |
| Mobile (iOS) | Keychain | Keychain |
| Mobile (Android) | EncryptedSharedPreferences | EncryptedSharedPreferences |

### PKCE Flow (for all public clients)

1. Client generates `code_verifier` (random 43-128 chars)
2. Client computes `code_challenge = BASE64URL(SHA256(code_verifier))`
3. Client initiates auth: `GET /authorize?response_type=code&code_challenge=<challenge>&code_challenge_method=S256`
4. User authenticates → server returns `authorization_code`
5. Client exchanges code: `POST /token` with `code_verifier`
6. Server verifies `SHA256(code_verifier) == stored code_challenge`
7. Server issues access_token + refresh_token

### Migration Strategy

Phase 1 (Week 1-2): Deploy OAuth server alongside existing session auth
Phase 2 (Week 3-4): Web app migrates to OAuth (behind feature flag)
Phase 3 (Week 5-6): Mobile apps launch with OAuth
Phase 4 (Week 7-8): Remove session cookie auth (breaking change for old clients)

### Backward Compatibility

During Phase 1-3, both auth methods work. API middleware checks:
1. If `Authorization: Bearer <token>` header present → validate JWT
2. If `connect.sid` cookie present → validate session (legacy)
3. If neither → 401 Unauthorized

## Security Requirements

- All tokens must be revocable (token revocation endpoint)
- Refresh token rotation: each use generates a new refresh token and invalidates the old one
- Refresh token reuse detection: if a rotated-out refresh token is used, invalidate ALL tokens for that user (potential theft)
- Rate limiting on token endpoint: 10 requests/minute per IP
- PKCE required for all public clients (no client_secret for mobile/SPA)
- CORS: only allow registered redirect URIs
        """.strip(),
        "comments": [
            # Comment 1 - Architecture review
            """Architecture review notes from the security team meeting:

Approved the overall approach with these modifications:

1. **Access token expiry**: 15 minutes is good for web, but for mobile we should support longer-lived access tokens (1 hour) since mobile networks are less reliable and we don't want users re-authenticating constantly. Add a `token_lifetime` parameter to the client registration.

2. **Refresh token rotation concerns**: Rotation on every use can cause issues with concurrent requests on mobile. If the app fires 3 API calls simultaneously and all get 401, they'll all try to refresh at the same time. Only the first one gets the new refresh token — the other two use the "old" token and trigger reuse detection, logging the user out.

   Solution: Add a **grace period** of 30 seconds. A rotated-out refresh token is still valid for 30 seconds after rotation. This handles concurrent refresh without compromising security.

3. **Token revocation**: Need to handle the "user changes password" scenario. When a password changes, ALL refresh tokens for that user must be revoked. Add a `token_generation` field to the user table — increment on password change, include in refresh token payload, validate on use.

4. **node-oidc-provider**: Good choice but make sure we pin the version. They had a breaking change in v7→v8 that changed the PKCE validation behavior. Pin to v8.x.

5. **Web SPA token storage**: Keeping access tokens in memory only means they're lost on page refresh. Two options:
   a) Silent auth using refresh token cookie (recommended)
   b) Use `sessionStorage` (less secure, survives refresh but not new tabs)

   Going with (a). The refresh token HttpOnly cookie + `/silent-refresh` iframe endpoint.""",

            # Comment 2 - Implementation progress
            """Sprint 1 complete. Here's what's deployed to staging:

### OAuth Server (`/auth/*` endpoints)
- `POST /auth/authorize` — authorization endpoint with PKCE
- `POST /auth/token` — token endpoint (authorization_code, refresh_token grants)
- `POST /auth/revoke` — token revocation
- `GET /auth/.well-known/openid-configuration` — OIDC discovery
- `GET /auth/jwks` — JSON Web Key Set for token verification

### Token Implementation
- Access tokens: RS256 JWT, signed with RSA-2048 key pair
- Key rotation: JWKS endpoint serves current + previous key, rotation every 90 days
- Refresh tokens: opaque UUID v4, stored in `refresh_tokens` PostgreSQL table
- Refresh token table schema:
```sql
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id),
    client_id VARCHAR(255) NOT NULL,
    token_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 of token
    previous_token_hash VARCHAR(64),  -- for rotation tracking
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    rotated_at TIMESTAMP,  -- set when token is rotated out
    revoked_at TIMESTAMP,
    user_agent TEXT,
    ip_address INET
);
```

### Dual Auth Middleware
Both cookie-session and Bearer token auth working simultaneously. Tested with:
- Existing web app (cookie) → works ✅
- Postman with Bearer token → works ✅
- Mobile prototype (iOS) → works ✅

### Outstanding
- Silent refresh iframe for SPA
- Refresh token grace period (30 second window)
- Reuse detection alerting
- Load testing
- Google OAuth social login integration with new flow""",

            # Comment 3 - QA testing results
            """QA testing results for the OAuth implementation:

**Functional Tests (47/47 passed):**
- Authorization code flow with PKCE ✅
- Token refresh ✅
- Token revocation ✅
- Invalid PKCE verifier rejection ✅
- Expired token rejection ✅
- Refresh token rotation ✅
- Concurrent refresh handling (grace period) ✅
- Password change → all tokens revoked ✅
- Dual auth (cookie + Bearer) ✅
- OIDC discovery endpoint ✅
- CORS enforcement ✅
- Rate limiting on /auth/token ✅

**Security Tests (12/12 passed):**
- PKCE downgrade attack (removing code_challenge) → rejected ✅
- Token injection (modified JWT) → rejected ✅
- Refresh token reuse (after grace period) → all tokens revoked ✅
- CSRF on token endpoint → rejected (no cookie-based auth on /token) ✅
- Open redirect via redirect_uri → rejected (whitelist enforced) ✅
- Token endpoint brute force → rate limited after 10 requests ✅

**Performance Tests:**
- Token issuance: 2.3ms average (P99: 8.1ms)
- Token validation (JWT verify): 0.4ms average (P99: 1.2ms)
- Refresh token rotation: 4.1ms average (P99: 12.3ms)
- Throughput: 2,400 token operations/second on single node

**One issue found:** The `token_generation` field check on refresh adds a database query on every refresh. For mobile clients refreshing every 15 minutes, this is ~96 queries/day per active user. With 45K MAU, that's ~4.3M queries/month just for generation checks. Suggest caching the generation value in Redis with 5-minute TTL.

Opening follow-up ticket for the Redis caching optimization.""",

            # Comment 4 - Migration execution
            """Phase 2 migration update (web app → OAuth):

Feature flag `use_oauth_auth` rolled out:
- Week 3: 5% of web users → 0 issues reported
- Week 4: 25% → 1 issue: silent refresh iframe blocked by Firefox Enhanced Tracking Protection. Fix: added `/.well-known/oauth` to the exception list in CSP headers.
- Week 5: 100% of web users on OAuth

Key metrics comparison (cookie vs OAuth):
- Login success rate: 99.2% → 99.4% (slight improvement due to fewer session expiry issues)
- API error rate: 0.08% → 0.09% (within normal variance)
- Average auth latency: 2ms → 3ms (JWT verification slightly slower than session lookup, acceptable)

Mobile launch (Phase 3) started Week 5. iOS app in TestFlight, Android in internal testing. Both using PKCE flow with Keychain/EncryptedSharedPreferences storage.

Phase 4 (remove cookie auth) scheduled for Week 9 — pushed back 1 week to give mobile users more time to update. Will require minimum app version check before removing the cookie fallback.""",

            # Comment 5 - Completion
            """All phases complete as of 2024-03-15.

Final state:
- OAuth 2.0 + PKCE is the sole authentication method
- Session cookie auth removed in v5.0.0 (PR #2198)
- express-session and Redis session store removed from dependencies
- 45,000 monthly active users migrated with 0 forced logouts

Post-migration metrics (30 days):
- Token issuance: 2.1M access tokens, 890K refresh tokens
- Refresh token rotation: working correctly, 0 reuse detection alerts
- Mobile app adoption: 12,400 iOS users, 8,200 Android users
- Auth-related support tickets: down 34% (fewer "I got logged out randomly" complaints)

Security audit by external firm completed 2024-03-10. No critical findings. One medium finding: refresh token grace period should be reduced from 30s to 10s. Updated in PR #2215.

Closing this ticket. Total effort: 8 weeks, 3 engineers."""
        ],
    },
]


# Ground truth QA pairs for long tickets
LONG_QA_PAIRS = [
    {
        "question": "What caused the 23-minute production database outage and what were the fixes?",
        "target_ticket": "LONG-101",
        "ground_truth": "The outage was caused by a PostgreSQL failover where the secondary node had a stale pg_hba.conf that didn't include authentication rules for three microservices deployed after the DR drill. When failover occurred, these services couldn't authenticate, causing cascading failures via shared PgBouncer pools. PgBouncer's server_login_retry=0 config meant it wouldn't retry failed logins. Fixes included: SSM-based pg_hba.conf sync between primary and secondary, PgBouncer server_login_retry set to 3, health checks updated to verify all service accounts. After fixes, controlled failover test showed 62-second total recovery (down from 23 minutes).",
    },
    {
        "question": "Why is PgBouncer not recovering automatically after the database failover?",
        "target_ticket": "LONG-101",
        "ground_truth": "PgBouncer's config had server_login_retry=0, which means it will NOT retry a failed login. Once the auth failure happened (due to stale pg_hba.conf on the failover node), the server connection was marked dead and PgBouncer needed a full restart to try again. The fix was setting server_login_retry=3 and adding server_reconnect_timeout=5 so PgBouncer retries auth failures automatically. This was implemented in PR #1247.",
    },
    {
        "question": "What's causing the memory leak in the order processing service and how was it fixed?",
        "target_ticket": "LONG-201",
        "ground_truth": "Three memory leaks were found: 1) OrderProcessor._processing_cache was a class-level dict that grew with every order (never cleaned up) — leaked ~2KB per order, ~6MB/hour. Fixed by making it an instance variable with cleanup in a finally block. 2) inventory_client.py _response_cache had the same class variable leak pattern — leaked ~1MB/hour. 3) utils/cache.py had a TTL cache using time.time() for expiry but the cleanup thread used time.monotonic(), so entries were never expired. Fixed by using time.monotonic() consistently. After fixes, memory growth dropped from 2.1MB/hour to 0.2MB/hour (normal Python heap fragmentation). Deployed as v4.2.2, PR #892.",
    },
    {
        "question": "How does the tiered API rate limiting work and what algorithm does it use?",
        "target_ticket": "LONG-301",
        "ground_truth": "Rate limiting is implemented as a custom Kong plugin with Redis backend using the sliding window log algorithm. Tiers: Free (60 req/min), Starter (300), Business (1000), Enterprise (5000). Redis stores sorted sets keyed by SHA-256 hash of the API key (not plain text, per security review). The plugin looks up the customer's plan via the billing service API (with 5-min cache). If Redis is unavailable, it falls back to in-memory limiting at 50% of plan limits after a 30-second open circuit. Enterprise customers can have custom limits. Latency impact: P50 +0.3ms, P99 +1.2ms.",
    },
    {
        "question": "What security issues were found in the rate limiter and how were they addressed?",
        "target_ticket": "LONG-301",
        "ground_truth": "Security review found: 1) [HIGH] API keys stored in plain text in Redis keys — fixed by using SHA-256 hash of the API key as the Redis key. 2) [MEDIUM] Billing service endpoint /internal/api-keys/{key}/plan had no authentication — fixed by adding mTLS using existing service mesh certificates. 3) [LOW] Rate limit headers expose exact limits — kept as-is (standard practice, documented in API docs). 4) [INFO] Circuit breaker 30-second open state too long during Redis Sentinel failover — reduced to 10 seconds with Sentinel SUBSCRIBE for failover detection.",
    },
    {
        "question": "Why are product updates silently disappearing from search results?",
        "target_ticket": "LONG-401",
        "ground_truth": "The Elasticsearch bulk indexing code only checked the top-level HTTP 200 status code and ignored individual document errors in the bulk response. About 3.2% of products (~1,600/day) were silently dropped. 78% of failures were 429 es_rejected_execution (ES thread pool full during peak hours), 15% were 409 version conflicts, 5% were 400 mapper_parsing_exception from bad UTF-8, 2% were 413 circuit breaker exceptions from oversized batches. The fix: parse individual item errors, retry 429/503 errors with exponential backoff using Celery's native retry, send permanent failures to a PostgreSQL dead letter queue with hourly retry job. After fix: 0% documents silently lost, 98.2% retry success rate.",
    },
    {
        "question": "How does the dead letter queue work for failed Elasticsearch documents?",
        "target_ticket": "LONG-401",
        "ground_truth": "Failed documents that can't be indexed after 3 retries are written to a search_indexing_failures PostgreSQL table with document_id, document_json (JSONB), error_type, error_detail, created_at, retried_at, resolved_at, and retry_count columns. A Celery beat task runs hourly to retry unresolved failures with exponential backoff, up to 10 total retries. In production, the DLQ retry resolved 67 of 89 failures after the nightly data cleanup fixed UTF-8 issues. The remaining 22 were genuinely corrupt product descriptions needing manual merchant intervention.",
    },
    {
        "question": "How does the OAuth 2.0 PKCE flow work for mobile authentication?",
        "target_ticket": "LONG-501",
        "ground_truth": "The PKCE flow: 1) Client generates code_verifier (random 43-128 chars), 2) Computes code_challenge = BASE64URL(SHA256(code_verifier)), 3) Initiates auth with GET /authorize including code_challenge and code_challenge_method=S256, 4) User authenticates, server returns authorization_code, 5) Client exchanges code via POST /token with code_verifier, 6) Server verifies SHA256(code_verifier) matches stored code_challenge, 7) Server issues access_token (JWT, RS256) + refresh_token (opaque UUID). Mobile stores tokens in Keychain (iOS) or EncryptedSharedPreferences (Android). Access tokens are 1 hour for mobile (15 min for web).",
    },
    {
        "question": "How are refresh tokens handled to prevent token theft?",
        "target_ticket": "LONG-501",
        "ground_truth": "Refresh tokens use rotation: each use generates a new refresh token and invalidates the old one. To handle concurrent mobile requests, there's a 30-second grace period where a rotated-out token remains valid (later reduced to 10 seconds per security audit). If a rotated-out token is used AFTER the grace period, ALL tokens for that user are revoked (reuse detection — indicates potential theft). Tokens include a token_generation field from the users table that's incremented on password change, so changing password revokes all refresh tokens. Tokens are stored as SHA-256 hashes in PostgreSQL, never in plain text.",
    },
    {
        "question": "What was the migration plan from session cookies to OAuth and how did it go?",
        "target_ticket": "LONG-501",
        "ground_truth": "Four phases: Phase 1 (Week 1-2) deployed OAuth server alongside session auth. Phase 2 (Week 3-5) migrated web app using feature flag (5% → 25% → 100%). One issue: Firefox Enhanced Tracking Protection blocked the silent refresh iframe, fixed with CSP headers. Phase 3 (Week 5+) launched mobile apps with PKCE. Phase 4 (Week 9) removed session cookie auth entirely in v5.0.0. Results: 45,000 users migrated with 0 forced logouts, auth-related support tickets down 34%, mobile adoption reached 12,400 iOS + 8,200 Android users. External security audit found one medium issue: grace period reduced from 30s to 10s.",
    },
]
