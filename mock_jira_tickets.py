"""
Mock Jira tickets for RAG experiment benchmarking.

Design principles:
- Tickets span a realistic e-commerce platform project ("ShopFlow")
- Clear semantic clusters so we can measure retrieval accuracy
- Ground truth query-to-ticket relevance pairs for evaluation
- Varying ticket quality (clean vs noisy descriptions, with/without comments)
- Realistic writing styles: lazy reporters, stack trace dumpers, vague PMs, etc.
"""

MOCK_TICKETS = [
    # ============================================================
    # CLUSTER 1: Payment Processing Issues
    # ============================================================
    {
        "key": "SHOP-101",
        "summary": "checkout 500 error stripe",
        "type": "Bug",
        "priority": "Critical",
        "status": "Open",
        "labels": ["payments", "stripe", "checkout"],
        "components": ["payment-service"],
        "description": """Getting this error in prod when ppl try to pay:

```
Traceback (most recent call last):
  File "/app/payment_service/stripe_client.py", line 142, in create_charge
    response = stripe.Charge.create(
  File "/usr/local/lib/python3.9/site-packages/stripe/api_resources/charge.py", line 38, in create
    return cls._static_request("post", url, params=params)
  File "/usr/local/lib/python3.9/site-packages/stripe/api_requestor.py", line 298, in request
    raise error.APIConnectionError(msg)
stripe.error.APIConnectionError: Request timed out after 30000ms
```

happens like ~15% of the time. started after we rotated API keys last thurs

env: prod
version: v2.4.1
""",
        "comments": [
            "I traced the issue to our Stripe API key rotation. The old key is being cached in Redis and used for ~15% of requests. We need to invalidate the cache after key rotation.",
            "Confirmed - Redis TTL for stripe_api_key is set to 24h but we rotate keys every 12h. Fix: reduce TTL to 6h or add cache bust on rotation.",
            "Hotfix deployed in v2.4.2. Monitoring shows 0 Stripe 500 errors in last 4hrs."
        ]
    },
    {
        "key": "SHOP-102",
        "summary": "PayPal refund fails silently for old orders (>60 days)",
        "type": "Bug",
        "priority": "High",
        "status": "In Progress",
        "labels": ["payments", "paypal", "refunds"],
        "components": ["payment-service"],
        "description": """Refund from admin panel is not working proper for PayPal orders that are more than 60 days old. The UI is showing "Refund Processed" but no money goes back to customer.

I check the PayPal API logs and see error code 10009 (Transaction refused) but our code is swallowing this error and marking refund as success anyway.

23 customers are affected, total amount $4,567.89

We need to:
1. Handle PayPal error code 10009 properly
2. Show error msg to admin when refund cant be processed
3. Maybe suggest manual bank transfer for old orders?

cc @finance-team
""",
        "comments": [
            "Adding proper error handling for PP error codes. PR #892 up for review.",
            "We should also add a pre-check b4 showing the refund btn - if order is >60 days, show a warning."
        ]
    },
    {
        "key": "SHOP-103",
        "summary": "Apple Pay support for mobile checkout",
        "type": "Story",
        "priority": "High",
        "status": "To Do",
        "labels": ["payments", "mobile", "apple-pay"],
        "components": ["payment-service", "mobile-app"],
        "description": """Users have been asking for Apple Pay for months. We're losing conversions on iOS because ppl dont want to type card numbers on mobile.

AC:
- Apple Pay btn on checkout for iOS w/ compatible devices
- All card types (Visa, MC, Amex)
- Shows as "Apple Pay" in order history
- Works w/ existing discount codes & promos
- 3DS auth when required

Tech notes:
- Use Stripe's Apple Pay integration (PassKit framework)
- Need Apple Merchant ID cert from iOS team
- Must follow Apple HIG for payment btns
""",
        "comments": [
            "Design mockups attached. Apple Pay btn will sit between card form and PayPal btn.",
            "We need the Apple Merchant ID cert from the iOS team first. I've pinged @sarah about it.",
            "bump - @sarah any update on the cert?",
            "@mike she's OOO til monday, I'll get it sorted next week"
        ]
    },
    {
        "key": "SHOP-104",
        "summary": "Implement payment retry logic with exponential backoff",
        "type": "Task",
        "priority": "Medium",
        "status": "In Progress",
        "labels": ["payments", "reliability", "backend"],
        "components": ["payment-service"],
        "description": """Currently when a payment API call fails due to network issues, we immediately return an error to the user. This is not ideal.

We should impl retry logic w/ exponential backoff:
- Retry up to 3x on transient errors (network timeouts, 502, 503)
- Do NOT retry on permanent errors (invalid card, insufficient funds, fraud)
- Backoff: 1s, 2s, 4s
- Add idempotency key to prevent dup charges
- Log all retry attempts

Should bring payment failure rate from 3.2% to <1%.
""",
        "comments": [
            "I'll use tenacity lib for retry logic. Handles exp backoff natively.",
            "Make sure we categorize Stripe error codes correctly. Their docs list which are retryable: https://stripe.com/docs/error-codes",
            "thx for the link, will review"
        ]
    },

    # ============================================================
    # CLUSTER 2: Search & Product Discovery
    # ============================================================
    {
        "key": "SHOP-201",
        "summary": "Search returns 0 results for misspelled queries - no fuzzy matching",
        "type": "Bug",
        "priority": "High",
        "status": "Open",
        "labels": ["search", "elasticsearch", "ux"],
        "components": ["search-service"],
        "description": """search is broken for typos. users get zero results. see screenshot.
""",
        "comments": [
            "Can you give examples of what queries are failing?",
            "yeah - \"bycicle\" returns nothing (should show bicycles), same with \"headfones\", \"samsnug\" etc. Our ES config doesnt have any fuzzy matching enabled at all.",
            "Looked at analytics - 12% of searches return 0 results and 40% of those have obvious misspellings that should match products.",
            "We need to add: 1) fuzzy matching w/ edit distance 2, 2) phonetic matching using ES phonetic analysis plugin, 3) synonym mapping for common product terms",
            "I've enabled fuzziness=AUTO in the match query. Uses edit distance 1 for 3-5 char terms, distance 2 for >5 chars.",
            "Should we also add a 'Did you mean...' suggestion? ES has a built-in suggest API.",
            "Good idea. I'll add that as a follow-up ticket SHOP-205."
        ]
    },
    {
        "key": "SHOP-202",
        "summary": "Product recommendation engine - collaborative filtering",
        "type": "Story",
        "priority": "Medium",
        "status": "To Do",
        "labels": ["search", "recommendations", "ml"],
        "components": ["search-service", "ml-pipeline"],
        "description": """We need to build a personalized product recommendation engine to drive cross-sell revenue. The current "trending products" widget is not personalized and has low engagement.

Business context: Our competitors all have personalized recs. Product team estimates a 15% increase in cross-sell revenue within 3 months if we do this right.<br><br>

Technical approach from ML team discussion (see meeting notes from 2/14):
- Collaborative filtering using matrix factorization (ALS algorithm) on user-item interaction matrix
- Train nightly on last 90 days of data
- Serve via REST API, need &lt;50ms p99 latency
- A/B test against current trending widget

Data inputs: items frequently bought together, similar users' purchase history, browsing history &amp; wishlist

This is a Q2 OKR so pls prioritize accordingly.
""",
        "comments": [
            "I recommend starting w/ implicit feedback (views, adds-to-cart, purchases) rather than explicit ratings since we don't have a rating system.",
            "We should also consider content-based filtering as a fallback for new users (cold start problem).",
            "+1 on the cold start concern. That's going to be a big issue for us."
        ]
    },
    {
        "key": "SHOP-203",
        "summary": "ES index rebuild takes 6 hours - way too slow",
        "type": "Task",
        "priority": "Medium",
        "status": "Open",
        "labels": ["search", "elasticsearch", "performance"],
        "components": ["search-service"],
        "description": """Full ES index rebuild takes 6 hrs for 2M products. This blocks deploys and means search results can be stale for half a day after catalog updates.

Bottlenecks:
1. Single-threaded bulk indexing (needs parallelization)
2. Heavy analyzers on all text fields during indexing
3. No incremental indexing - full rebuild every time
4. refresh_interval=1s during bulk ops (should be -1)

Target: <1hr for full rebuild + incremental indexing for real-time updates.
""",
        "comments": [
            "Switching to parallel bulk indexing w/ 4 workers brought it down to 2.5hrs. Still need incremental indexing tho."
        ]
    },
    {
        "key": "SHOP-204",
        "summary": "Faceted search filters for product attributes",
        "type": "Story",
        "priority": "Medium",
        "status": "In Progress",
        "labels": ["search", "ux", "frontend"],
        "components": ["search-service", "web-frontend"],
        "description": """As a shopper I want to filter search results by size, color, brand, price range, rating etc so I can find stuff faster.

Reqs:
- Dynamic facets based on current search results (NOT static)
- Multi-select within facet (e.g. both "Red" AND "Blue")
- Show product count per facet value
- URL updates w/ filter state (shareable/bookmarkable)
- Mobile friendly collapsible panel

Tech: ES aggregations for facet counts, post_filter for multi-select behavior.
""",
        "comments": [
            "ES aggregation query working. FE PR #901 up with filter UI.",
            "We need to handle the case where a facet value has 0 results after applying other filters - hide it or show greyed out?",
            "Show greyed out w/ (0) count so users understand why their option isn't clickable."
        ]
    },

    # ============================================================
    # CLUSTER 3: User Authentication & Security
    # ============================================================
    {
        "key": "SHOP-301",
        "summary": "2FA implementation (TOTP) for user accounts",
        "type": "Story",
        "priority": "High",
        "status": "In Progress",
        "labels": ["security", "authentication", "2fa"],
        "components": ["auth-service"],
        "description": """Implement two-factor authentication using TOTP (Time-based One-Time Password). This is a compliance requirement and has been requested by multiple enterprise customers.

Requirements:
- Support TOTP compatible with Google Authenticator, Authy, etc.
- QR code setup flow w/ manual key entry fallback
- 10 single-use backup recovery codes
- "Remember this device" option (skip 2FA for 30 days on trusted devices)
- *Mandatory* 2FA for all admin accounts

Technical implementation:
- {{pyotp}} library for TOTP generation/verification
- Store encrypted TOTP secret in user table
- Recovery codes stored as bcrypt hashes
- Rate limit verification endpoint to prevent brute force
""",
        "comments": [
            "PR #876 has the BE impl. TOTP verification working w/ Google Authenticator.",
            "We need to add rate limiting to the 2FA verification endpoint. Max 5 attempts/min to prevent brute force.",
            "Added rate limiting in PR #879. Also added account lockout after 10 consecutive failed 2FA attempts."
        ]
    },
    {
        "key": "SHOP-302",
        "summary": "JWT tokens still valid after password change - SECURITY ISSUE",
        "type": "Bug",
        "priority": "Critical",
        "status": "Open",
        "labels": ["security", "authentication", "jwt"],
        "components": ["auth-service"],
        "description": """Security issue reported by penetration testing team. JWT tokens are not invalidated when user changes password.
""",
        "comments": [
            "Can someone add more detail here? What's the actual impact?",
            "If someone's acct gets compromised and they change their password, the attacker's JWT is STILL valid for up to 24hrs (our JWT expiry). There's also no way to 'log out all devices'. This is pretty bad.",
            "Fix options:\nA) token version/generation field in user tbl, increment on pw change, include in JWT, validate each request\nB) Redis blacklist of invalidated tokens\nC) reduce JWT expiry to 15min + use refresh tokens\n\nOption A is simplest, no new infra needed",
            "Going with option A. Adding `token_generation` col to users table. Migration PR #885.",
            "We should also add a 'Log out all sessions' button on the security settings page.",
            "same here - this came up in the security audit too"
        ]
    },
    {
        "key": "SHOP-303",
        "summary": "Rate limiting on /api/auth/login is broken",
        "type": "Bug",
        "priority": "High",
        "status": "Open",
        "labels": ["security", "rate-limiting", "api"],
        "components": ["auth-service", "api-gateway"],
        "description": """The rate limiter on login endpoint is using X-Forwarded-For header for IP identification but our LB overwrites this header. So ALL requests look like they come from same IP (the LB's IP).

Result: legit users get rate limited while attackers from different IPs arent individually limited. Basically the rate limiter is doing the opposite of what we want lol

Need to:
1. Configure LB to preserve original client IP
2. Use X-Real-IP header set by our nginx reverse proxy instead
3. Add account-based rate limiting (not just IP) to prevent credential stuffing
4. Progressive delays: 1s after 3 fails, 5s after 5, lockout after 10
""",
        "comments": [
            "X-Real-IP header is correct one to use. Updated rate limiter config.",
            "Also adding account-based limiting using Redis sorted sets to track failed attempts per username."
        ]
    },
    {
        "key": "SHOP-304",
        "summary": "OAuth2 social login - Google and GitHub",
        "type": "Story",
        "priority": "Medium",
        "status": "To Do",
        "labels": ["authentication", "oauth", "social-login"],
        "components": ["auth-service", "web-frontend"],
        "description": """Users keep asking for social login. Nobody wants to create another username/password. We need "Sign in with Google" and "Sign in with GitHub" on the login/register page.

Requirements:
- Google + GitHub btns on login/register
- Link social acct to existing acct if email matches
- User can unlink social login & set pw later
- Import profile photo from social acct
- Edge case: user signs up w/ email, then tries Google login w/ same email

Tech approach:
- OAuth2 authorization code flow
- Google: Google Identity Services lib
- GitHub: GitHub OAuth App
- New social_accounts table: provider + provider_user_id
""",
        "comments": [
            "Do we also want Apple Sign In? It's required for iOS apps that offer other social login options.",
            "Good point. Let's add Apple as fast-follow. The architecture supports adding new providers easily.",
            "@tom can you check if our Apple Developer acct has Sign In with Apple enabled?"
        ]
    },

    # ============================================================
    # CLUSTER 4: Performance & Infrastructure
    # ============================================================
    {
        "key": "SHOP-401",
        "summary": "product pages super slow on mobile",
        "type": "Bug",
        "priority": "High",
        "status": "Open",
        "labels": ["performance", "mobile", "frontend"],
        "components": ["web-frontend"],
        "description": """PDP is taking forever to load on mobile. PM is escalating this.

Lighthouse numbers are terrible.
""",
        "comments": [
            "Can you share the actual lighthouse scores? 'terrible' doesn't help us prioritize",
            "sorry - LCP: 4.2s (target <2.5s), FID: 180ms (target <100ms), CLS: 0.25 (target <0.1). This is on 3G throttling.",
            "ok so the main issues I see:\n1. Hero product image is 2.4MB unoptimized JPEG - should be WebP w/ lazy loading\n2. All product reviews (up to 500) load on initial render\n3. Third-party scripts (analytics, chat widget) block main thread for 800ms\n4. No SSR - entire page is CSR React",
            "Converting images to WebP with srcset for responsive sizes. This alone should save 70% bandwidth.",
            "Implementing virtual scrolling for reviews - only render first 10, load more on scroll.",
            "Moving analytics + chat widget to load after onload event. Using requestIdleCallback for non-critical scripts."
        ]
    },
    {
        "key": "SHOP-402",
        "summary": "DB connection pool exhausted during Black Friday flash sale - POSTMORTEM",
        "type": "Bug",
        "priority": "Critical",
        "status": "Open",
        "labels": ["performance", "database", "scaling"],
        "components": ["backend-core", "database"],
        "description": """Postmortem for the Black Friday incident. The PostgreSQL connection pool was exhausted within 5 minutes of the flash sale starting, causing cascading failures across ALL services.

*Timeline:*
- 00:00 - Sale starts, traffic spikes to 50K concurrent users
- 00:02 - Connection pool (max 100) fully utilized
- 00:05 - Connection queue backlog hits 500, requests start timing out
- 00:08 - Auto-scaler adds new pods but they can't get DB connections either (made it worse)
- 00:12 - Manual intervention: increased pool to 200, killed stale connections

*Root cause:* Long-running inventory check queries (avg 2.3s) hold connections while waiting for row-level locks on popular items.

*Action items:*
1. Deploy PgBouncer for transaction-level connection pooling
2. Optimize inventory queries with SELECT FOR UPDATE SKIP LOCKED
3. Add Redis-based inventory cache w/ write-through invalidation
4. Implement circuit breaker pattern for DB calls
""",
        "comments": [
            "PgBouncer deployed in front of PostgreSQL. Initial tests show it handles 1000 concurrent connections w/ only 50 actual DB connections. Huge improvement.",
            "Inventory query optimization reduced avg from 2.3s to 0.15s. SKIP LOCKED is a game changer for high-contention scenarios.",
            "We should do a load test before next sale. I'll set up k6 scripts.",
            "definately agree. last time was a disaster and we cant afford that again"
        ]
    },
    {
        "key": "SHOP-403",
        "summary": "Redis caching layer for product catalog API",
        "type": "Task",
        "priority": "Medium",
        "status": "To Do",
        "labels": ["performance", "caching", "redis"],
        "components": ["backend-core"],
        "description": """Product catalog API is hitting PG directly for every single request. With 10K RPM on the product detail endpoint this is putting way too much load on the DB.

Caching strategy:
- Cache product details in Redis, 5min TTL
- Cache-aside pattern: check Redis -> fallback to DB -> populate cache on miss
- Invalidate on product update (publish event from admin svc)
- Key format: {{product:{id}:v{version}}}
- Use Redis pipeline for batch fetches (category pages)

Expected: ~80% cache hit rate, reducing DB load by ~8K QPM.
""",
        "comments": [
            "Should we use Redis Cluster or single instance? Product catalog is ~500MB serialized.",
            "Single instance is fine for now. Redis handles 500MB no problem. Cluster later if needed.",
            "thx, will start impl this week"
        ]
    },
    {
        "key": "SHOP-404",
        "summary": "K8s HPA setup for all API services",
        "type": "Task",
        "priority": "Medium",
        "status": "In Progress",
        "labels": ["infrastructure", "kubernetes", "scaling"],
        "components": ["devops"],
        "description": """Configure Horizontal Pod Autoscaler for all API svcs to handle traffic spikes.

Config:
| Service | Min | Max | CPU Target |
| payment-service | 3 | 15 | 70% |
| search-service | 2 | 20 | 60% (memory-intensive) |
| auth-service | 2 | 10 | 75% |
| product-service | 3 | 25 | 65% |

Also need:
- PDBs for availability during scaling
- Custom metrics adapter for scaling on request latency (not just CPU)
- Pre-warming: scale up 30min before scheduled sales
""",
        "comments": [
            "HPA manifests ready for all services. Testing w/ load generator before applying to prod.",
            "We should also add VPA (Vertical Pod Autoscaler) recommendations to right-size resource requests."
        ]
    },

    # ============================================================
    # CLUSTER 5: Order Management & Shipping
    # ============================================================
    {
        "key": "SHOP-501",
        "summary": "FedEx webhook status mapping is wrong",
        "type": "Bug",
        "priority": "High",
        "status": "Open",
        "labels": ["orders", "shipping", "webhooks"],
        "components": ["order-service"],
        "description": """FedEx shipping webhooks are recieved by our system but the status mapping in the webhook handler is completely wrong:

Current (WRONG):
{code}
"IT" (In Transit) -> "Processing"  # should be "Shipped"
"DL" (Delivered) -> "Shipped"      # should be "Delivered"
"DE" (Delivery Exception) -> ???   # not handled at all, should be "Delivery Issue"
{code}

Customers seeing stale statuses -> support getting flooded with "where is my order" tickets.

Also: the webhook endpoint has NO signature verification. Anyone could send fake status updates. This is a security issue too.
""",
        "comments": [
            "Fixed status mapping in PR #910. Also added FedEx webhook signature verification using their HMAC key.",
            "Need to backfill ~340 orders with incorrect statuses. Writing migration script now.",
            "Migration complete. Affected orders show correct status. Customer notification emails sent."
        ]
    },
    {
        "key": "SHOP-502",
        "summary": "Order splitting for multi-warehouse fulfillment",
        "type": "Story",
        "priority": "Medium",
        "status": "To Do",
        "labels": ["orders", "shipping", "fulfillment"],
        "components": ["order-service", "inventory-service"],
        "description": """Right now all items in an order have to ship from same warehouse. This causes delays when some items are only in stock at a warehouse thats far from the customer.

We need to automatically split orders when items are across multiple warehouses and pick the optimal warehouse for each item based on proximity to delivery addr.

Reqs:
- Auto-split orders across warehouses
- Optimal warehouse selection based on distance
- Seperate tracking numbers per shipment
- Consolidate shipping charges (dont charge more for splits)
- Handle partial cancellations within split orders

Tech: new order_shipments tbl, warehouse mgmt API integration for real-time inventory by location, distance-based routing for shipping cost optimization
""",
        "comments": [
            "This is a big one. I suggest we break into phases:\nPhase 1 - split logic + warehouse selection\nPhase 2 - shipping cost optimization\nPhase 3 - partial cancellation handling",
            "Agreed. Let's start w/ Phase 1. I'll create sub-tasks."
        ]
    },
    {
        "key": "SHOP-503",
        "summary": "order emails not sending",
        "type": "Bug",
        "priority": "High",
        "status": "Open",
        "labels": ["orders", "notifications", "email"],
        "components": ["notification-service"],
        "description": """customers not getting emails when order status changes. started 3 days ago.
""",
        "comments": [
            "@devops-team can someone look at this? customers are complaning",
            "I checked - order status change events ARE being published to RabbitMQ and notification service IS consuming them. But SendGrid API calls are failing with 403 Forbidden.",
            "Found it. SendGrid API key was rotated last week but new key was only updated in staging, not prod. Classic.",
            "Prod SendGrid key updated. Emails flowing again.",
            "I'm queuing up the missed notifications for last 3 days. About 2,400 order status emails to resend.",
            "All backlogged emails sent. Creating follow-up ticket for moving secrets to HashiCorp Vault so this never happens again."
        ]
    },

    # ============================================================
    # CLUSTER 6: Inventory Management
    # ============================================================
    {
        "key": "SHOP-601",
        "summary": "overselling bug - race condition in inventory service",
        "type": "Bug",
        "priority": "Critical",
        "status": "Open",
        "labels": ["inventory", "concurrency", "database"],
        "components": ["inventory-service"],
        "description": """We have a race condition that allows overselling. Two concurrent requests can both read the same inventory count, both see sufficient stock, and both decrement. Result: negative inventory.

Example: SKU-1234 had 1 unit. Two simultaneous purchases both succeeded -> inventory count is now -1. This has happened 47 times in the last month. We've had to manually cancel orders and apologize to customers.

The problem is the check-and-decrement is not atomic:

{code:python}
count = db.query("SELECT count FROM inventory WHERE sku = %s", sku)
if count > 0:
    db.execute("UPDATE inventory SET count = count - 1 WHERE sku = %s", sku)
{code}

Fix should use atomic UPDATE w/ WHERE clause:

{code:python}
result = db.execute(
    "UPDATE inventory SET count = count - 1 WHERE sku = %s AND count > 0", sku
)
if result.rowcount == 0:
    raise OutOfStockError()
{code}
""",
        "comments": [
            "Atomic UPDATE fix is deployed. Also added DB constraint: ALTER TABLE inventory ADD CONSTRAINT positive_count CHECK (count >= 0)",
            "For flash sales we should consider using Redis for inventory reservation with TTL-based expiry for unpaid reservations.",
            "+1, the DB approach wont scale for Black Friday traffic"
        ]
    },
    {
        "key": "SHOP-602",
        "summary": "Low stock alerts and automatic reorder",
        "type": "Story",
        "priority": "Medium",
        "status": "To Do",
        "labels": ["inventory", "alerts", "automation"],
        "components": ["inventory-service", "notification-service"],
        "description": """The inventory team needs better visibility into stock levels. Currently they manually check a spreadsheet which is not scalable and we keep running out of popular items.

As an inventory manager I want alerts when stock falls below a threshold and automatic reorder suggestions so we never run out.

*Requirements:*
- Configurable low-stock threshold per product (default 10 units)
- Email + Slack alert when below threshold
- Dashboard: all low-stock items sorted by days-until-stockout
- Auto reorder suggestion based on avg daily sales rate
- Phase 2: supplier API integration for automated PO creation

*Success metric:* reduce stockout incidents by 80%
""",
        "comments": [
            "I'll start w/ alerting. Can use existing RabbitMQ setup to publish low_stock events.",
            "For days-until-stockout calc we should use 30-day rolling avg of daily sales, weighted toward recent days."
        ]
    },

    # ============================================================
    # CLUSTER 7: API & Integration Issues
    # ============================================================
    {
        "key": "SHOP-701",
        "summary": "API response times tanked after audit logging middleware deploy",
        "type": "Bug",
        "priority": "High",
        "status": "Open",
        "labels": ["api", "performance", "logging"],
        "components": ["api-gateway"],
        "description": """After adding audit logging middleware in v3.2.0, ALL API endpoints got 150-200ms slower. The middleware is doing a synchronous INSERT into the audit_log table on every single request. This is in the hot path.

Before v3.2.0: p50=45ms, p95=120ms, p99=250ms
After v3.2.0:  p50=195ms, p95=320ms, p99=480ms

Thats a 3-4x regression on p50. Not acceptable.

Options to fix:
1. Fire-and-forget async (risk: lost logs on crash)
2. Publish to msg queue, consume async (preferred)
3. Buffer in memory, flush in batches every 5s

pls prioritize this, its affecting all customers
""",
        "comments": [
            "Going with option 2 - publishing to RabbitMQ. Seperate consumer writes to audit_log table.",
            "This brought p50 back down to 52ms. Async consumer processes logs w/ ~1s delay which is fine for audit."
        ]
    },
    {
        "key": "SHOP-702",
        "summary": "N+1 query issue on product listing - GraphQL",
        "type": "Bug",
        "priority": "High",
        "status": "In Progress",
        "labels": ["api", "graphql", "performance", "database"],
        "components": ["api-gateway", "backend-core"],
        "description": """The GraphQL product listing endpoint has a classic N+1 problem. Fetching 20 products w/ categories and reviews executes:

- 1 query for products
- 20 queries for categories (1 per product)
- 20 queries for review aggregates (1 per product)

= 41 queries for a single page load. Avg 800ms.

Solution: DataLoader pattern to batch + cache DB queries within a request.

With DataLoader:
- 1 query for products
- 1 batched query for all 20 categories
- 1 batched query for all 20 review aggregates

= 3 queries total. Expected <100ms.
""",
        "comments": [
            "Implemented DataLoader for categories. Working on reviews next.",
            "Both DataLoaders in place. Response dropped from 800ms to 85ms. PR #920 up for review."
        ]
    },

    # ============================================================
    # CLUSTER 8: Frontend / UI Issues
    # ============================================================
    {
        "key": "SHOP-801",
        "summary": "cart items disappear switching between phone and laptop",
        "type": "Bug",
        "priority": "High",
        "status": "Open",
        "labels": ["frontend", "cart", "sync"],
        "components": ["web-frontend", "mobile-app"],
        "description": """When user is logged in and adds stuff to cart on phone then opens laptop, cart is different. Its like they have two seperate carts.

please fix, customers keep complaining about this to support. we get like 10-15 tickets a week about it.
""",
        "comments": [
            "Looked into it - root cause is that mobile app stores cart in localStorage while web uses server-side cart API. Neither syncs w/ the other.",
            "Fix plan:\n1. Make both mobile + web use server-side cart API\n2. On app open, sync local cart to server (merge: union of items, max qty)\n3. Add realtime cart sync via WebSocket for multi-device\n4. Conflict resolution for simultaneous edits",
            "The server-side cart API already exists, mobile just isn't using it. Migrating mobile to the API.",
            "What about offline support? Mobile users might add items w/o internet.",
            "Good point. We'll keep localStorage as a write-ahead log. Sync to server when connection restored."
        ]
    },
    {
        "key": "SHOP-802",
        "summary": "Accessibility audit failures - checkout flow WCAG 2.1 AA",
        "type": "Bug",
        "priority": "High",
        "status": "To Do",
        "labels": ["frontend", "accessibility", "compliance"],
        "components": ["web-frontend"],
        "description": """External accessibility audit found 23 WCAG 2.1 AA violations in the checkout flow. Full report attached (audit_report_2024_q1.pdf).

*Critical (must fix before March deadline):*
- Form inputs missing associated labels (8 instances)
- Color contrast ratio below 4.5:1 on error msgs (red #FF0000 on white)
- No keyboard navigation for payment method selection (radio btns)
- Screen reader cannot identify current checkout step

*Major (should fix):*
- Focus not moved to error summary after form validation failure
- Timeout warning for payment session not accessible
- Order summary table missing proper header associations

Legal has flagged this as a compliance risk. March deadline is firm.
""",
        "comments": [
            "Starting with form label fixes - easiest wins first.",
            "For color contrast issue, proposing we change error red from #FF0000 to #D32F2F which passes AA on white backgrounds.",
            "can someone share the full audit pdf? I dont have access to the shared drive",
            "uploaded to confluence: [checkout accessibility audit|https://confluence.internal/display/SHOP/Checkout+A11y+Audit]"
        ]
    },

    # ============================================================
    # CLUSTER 9: Data & Analytics
    # ============================================================
    {
        "key": "SHOP-901",
        "summary": "conversion rate metric is wrong on analytics dashboard",
        "type": "Bug",
        "priority": "Medium",
        "status": "Open",
        "labels": ["analytics", "dashboard", "data"],
        "components": ["analytics-service"],
        "description": """conversion rate on dashboard looks way too high. showing 12.5% but that cant be right. pls investigate.
""",
        "comments": [
            "yeah 12.5% is definitely wrong, industry avg for ecommerce is 2-4%. our actual is probably around 3.2%",
            "Found it - the formula in analytics_service/reports.py line 234 is wrong.\n\nCurrent (wrong): conversions / unique_sessions_with_purchase\nThis divides purchases by purchasers which is basically always ~1\n\nCorrect: unique_sessions_with_purchase / total_unique_sessions\n\nNumerator and denominator are swapped lol",
            "Embarrassing. Fixed in PR #930. Also added unit tests for ALL metric calcs to prevent this.",
            "We should add data validation alerts - if any metric changes by >50% day-over-day, trigger alert.",
            "@analytics-team heads up the dashboard numbers are going to drop significantly after the fix deploys. This is expected."
        ]
    },
    {
        "key": "SHOP-902",
        "summary": "Real-time event streaming pipeline - move from batch to Kafka",
        "type": "Story",
        "priority": "Medium",
        "status": "To Do",
        "labels": ["analytics", "kafka", "infrastructure", "streaming"],
        "components": ["analytics-service", "devops"],
        "description": """Our analytics pipeline is batch-based and always 24hrs stale. We need to move to real-time event streaming with Kafka.

*Current state:* events collected in log files -> nightly cron job processes them -> loads into analytics DB. Dashboards always a day behind.

*Target state:*
- All user events (page views, clicks, purchases) published to Kafka topics
- Stream processing w/ Kafka Streams for real-time aggregations
- Real-time dashboard updates via WebSocket
- Event replay for reprocessing historical data

*Architecture:*
- 3 Kafka brokers, 2 ZK nodes
- Topics: user_events, order_events, inventory_events
- Retention: 7 days raw events, aggregated data in PG
""",
        "comments": [
            "Should we consider Kafka alternatives? AWS Kinesis or Redpanda are both simpler to operate.",
            "Let's stick w/ Kafka - team has experience. Can use Confluent Cloud to reduce ops overhead.",
            "makes sense. when do we want to kick this off? its a pretty big project"
        ]
    },

    # ============================================================
    # CLUSTER 10: DevOps & CI/CD
    # ============================================================
    {
        "key": "SHOP-1001",
        "summary": "CI/CD pipeline is 45 min - devs complaining",
        "type": "Task",
        "priority": "High",
        "status": "In Progress",
        "labels": ["devops", "ci-cd", "performance"],
        "components": ["devops"],
        "description": """github actions pipeline takes 45 min for full run. devs are not happy. this is killing velocity.

breakdown:
- deps install: 8 min (no caching!!)
- unit tests: 12 min (running sequentially lol)
- integration tests: 15 min (spinning up docker each time)
- build: 5 min
- staging deploy: 5 min

optimization plan:
1. cache node_modules + pip (-8 min)
2. parallel unit tests across 4 workers (-9 min)
3. persistent docker svcs in CI (-10 min)
4. turbo/nx for incremental builds (-5 min)

target: <15 min
""",
        "comments": [
            "Caching impl'd. Pipeline is now 37 min.",
            "Parallel test workers brought it to 28 min. Docker optimization next.",
            "any ETA on getting this under 20? its still painful",
            "hoping by end of sprint. the docker service caching is trickier than expected"
        ]
    },

    # ============================================================
    # Additional tickets for noise/diversity
    # ============================================================
    {
        "key": "SHOP-1101",
        "summary": "Update README with new development setup instructions",
        "type": "Task",
        "priority": "Low",
        "status": "Done",
        "labels": ["documentation"],
        "components": ["devops"],
        "description": """README is outdated. Needs to cover:
- new docker-compose setup for local dev
- env var requirements
- how to run tests
- API docs links

pls update thx
""",
        "comments": [
            "Done. PR #940 merged."
        ]
    },
    {
        "key": "SHOP-1102",
        "summary": "React upgrade v17 -> v18",
        "type": "Task",
        "priority": "Medium",
        "status": "Done",
        "labels": ["frontend", "dependencies", "react"],
        "components": ["web-frontend"],
        "description": """Upgrade React from 17 to 18 for:
- Concurrent rendering
- Automatic batching
- useTransition / useDeferredValue hooks
- Streaming SSR

Migration:
1. Update react + react-dom to v18
2. Replace ReactDOM.render -> createRoot
3. Update tests for createRoot
4. Fix StrictMode double-render warnings
5. Full regression test
""",
        "comments": [
            "Upgrade complete. All tests passing, no regressions in QA.",
            "We should use startTransition for the product search input to prevent UI jank during search.",
            "+1 good idea"
        ]
    },
    {
        "key": "SHOP-1103",
        "summary": "CSRF protection missing on state-changing endpoints",
        "type": "Task",
        "priority": "High",
        "status": "In Progress",
        "labels": ["security", "api", "csrf"],
        "components": ["api-gateway", "auth-service"],
        "description": """Our API uses JWT auth but has no CSRF protection. JWT in headers is inherently CSRF-safe, BUT our mobile app falls back to cookie-based sessions when token refresh fails. This means POST/PUT/DELETE endpoints are vulnerable to CSRF when cookie session is active.

Implementation plan:
- Generate CSRF token on session creation, store in cookie + response header
- Validate CSRF token on all non-GET requests when cookie auth is used
- SameSite=Strict cookie attribute as defense-in-depth
- Skip CSRF check for requests w/ valid JWT in Authorization header

This was flagged in the security audit last month. Need to fix ASAP.
""",
        "comments": [
            "Implementing double-submit cookie pattern. CSRF token generated as signed JWT.",
            "We need to update mobile app to send CSRF token header on all POST/PUT/DELETE requests.",
            "@mobile-team pls review the integration guide I put on confluence"
        ]
    },
]

# ============================================================
# Ground truth: query -> relevant ticket keys
# These are the "golden" evaluation pairs
# ============================================================
GROUND_TRUTH_QUERIES = [
    {
        "query": "payment processing failing with errors",
        "relevant_keys": ["SHOP-101", "SHOP-102", "SHOP-104"],
        "primary_key": "SHOP-101",  # Most relevant
        "category": "semantic"
    },
    {
        "query": "Stripe API 500 error during checkout",
        "relevant_keys": ["SHOP-101"],
        "primary_key": "SHOP-101",
        "category": "specific"
    },
    {
        "query": "how to add new payment method like Apple Pay",
        "relevant_keys": ["SHOP-103"],
        "primary_key": "SHOP-103",
        "category": "semantic"
    },
    {
        "query": "refund not working for old orders",
        "relevant_keys": ["SHOP-102"],
        "primary_key": "SHOP-102",
        "category": "semantic"
    },
    {
        "query": "search returning no results for typos and misspellings",
        "relevant_keys": ["SHOP-201", "SHOP-204"],
        "primary_key": "SHOP-201",
        "category": "semantic"
    },
    {
        "query": "Elasticsearch performance slow indexing",
        "relevant_keys": ["SHOP-203"],
        "primary_key": "SHOP-203",
        "category": "specific"
    },
    {
        "query": "product recommendation system machine learning",
        "relevant_keys": ["SHOP-202"],
        "primary_key": "SHOP-202",
        "category": "semantic"
    },
    {
        "query": "two factor authentication setup",
        "relevant_keys": ["SHOP-301"],
        "primary_key": "SHOP-301",
        "category": "semantic"
    },
    {
        "query": "security vulnerability with session tokens after password reset",
        "relevant_keys": ["SHOP-302", "SHOP-301", "SHOP-303"],
        "primary_key": "SHOP-302",
        "category": "semantic"
    },
    {
        "query": "login endpoint being brute forced",
        "relevant_keys": ["SHOP-303", "SHOP-301"],
        "primary_key": "SHOP-303",
        "category": "semantic"
    },
    {
        "query": "social login OAuth Google GitHub",
        "relevant_keys": ["SHOP-304"],
        "primary_key": "SHOP-304",
        "category": "specific"
    },
    {
        "query": "website is slow on mobile devices",
        "relevant_keys": ["SHOP-401", "SHOP-801"],
        "primary_key": "SHOP-401",
        "category": "semantic"
    },
    {
        "query": "database connection issues under high load",
        "relevant_keys": ["SHOP-402"],
        "primary_key": "SHOP-402",
        "category": "semantic"
    },
    {
        "query": "Redis caching strategy for products",
        "relevant_keys": ["SHOP-403"],
        "primary_key": "SHOP-403",
        "category": "specific"
    },
    {
        "query": "Kubernetes autoscaling configuration",
        "relevant_keys": ["SHOP-404"],
        "primary_key": "SHOP-404",
        "category": "specific"
    },
    {
        "query": "shipping status not updating for customers",
        "relevant_keys": ["SHOP-501", "SHOP-503"],
        "primary_key": "SHOP-501",
        "category": "semantic"
    },
    {
        "query": "orders shipping from multiple warehouses",
        "relevant_keys": ["SHOP-502"],
        "primary_key": "SHOP-502",
        "category": "semantic"
    },
    {
        "query": "customers not receiving email notifications",
        "relevant_keys": ["SHOP-503"],
        "primary_key": "SHOP-503",
        "category": "semantic"
    },
    {
        "query": "products being sold when out of stock overselling",
        "relevant_keys": ["SHOP-601"],
        "primary_key": "SHOP-601",
        "category": "semantic"
    },
    {
        "query": "inventory running low automatic alerts reorder",
        "relevant_keys": ["SHOP-602", "SHOP-601"],
        "primary_key": "SHOP-602",
        "category": "semantic"
    },
    {
        "query": "API response times slow after new middleware",
        "relevant_keys": ["SHOP-701"],
        "primary_key": "SHOP-701",
        "category": "semantic"
    },
    {
        "query": "N+1 query problem GraphQL",
        "relevant_keys": ["SHOP-702"],
        "primary_key": "SHOP-702",
        "category": "specific"
    },
    {
        "query": "shopping cart not syncing across devices",
        "relevant_keys": ["SHOP-801"],
        "primary_key": "SHOP-801",
        "category": "semantic"
    },
    {
        "query": "accessibility WCAG compliance issues",
        "relevant_keys": ["SHOP-802"],
        "primary_key": "SHOP-802",
        "category": "specific"
    },
    {
        "query": "analytics dashboard showing wrong numbers",
        "relevant_keys": ["SHOP-901"],
        "primary_key": "SHOP-901",
        "category": "semantic"
    },
    {
        "query": "real-time event processing Kafka streaming",
        "relevant_keys": ["SHOP-902"],
        "primary_key": "SHOP-902",
        "category": "specific"
    },
    {
        "query": "CI/CD pipeline too slow",
        "relevant_keys": ["SHOP-1001"],
        "primary_key": "SHOP-1001",
        "category": "semantic"
    },
    {
        "query": "CSRF cross-site request forgery protection",
        "relevant_keys": ["SHOP-1103", "SHOP-303"],
        "primary_key": "SHOP-1103",
        "category": "specific"
    },
    {
        "query": "how to handle concurrent database updates race condition",
        "relevant_keys": ["SHOP-601", "SHOP-402"],
        "primary_key": "SHOP-601",
        "category": "semantic"
    },
    {
        "query": "webhook integration not processing correctly",
        "relevant_keys": ["SHOP-501"],
        "primary_key": "SHOP-501",
        "category": "semantic"
    },
]
