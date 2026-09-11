# Security Policy

We take security vulnerabilities seriously. If you discover a security vulnerability, please follow these steps:

1. Do NOT open a public GitHub issue.
2. Do NOT discuss the vulnerability publicly until it has been addressed.
3. Email security concerns to hayder@alumni.harvard.edu.
   - Include a detailed description of the vulnerability.
   - Include steps to reproduce when applicable.
   - Include a potential impact assessment.
   - Include a suggested fix when available.

### response timeline

- Initial response within 2 business days.
- Status update within 7 business days.
- Resolution depends on severity and complexity.

### disclosure policy

- We will acknowledge receipt of your report within 48 hours.
- We will provide regular updates on the status of the vulnerability.
- Once the vulnerability is fixed, we will:
  - Credit you in the security advisory when you wish.
  - Publish a security advisory with details.

## security best practices

When deploying orla:

- Keep orla updated to the latest version.
- Run orla behind an authenticating reverse proxy such as nginx, Cloudflare, or a service mesh. Orla itself does not enforce authentication on its API.
- Use Postgres credentials with the minimum privileges orla requires. Read and write on its own database is enough, and no superuser is needed.
- Protect the upstream LLM and tool API keys orla holds in environment variables. Restrict who can read the process environment, and avoid committing `.env` files.
- Monitor `/metrics` and structured logs for anomalous request patterns. Every write to the control plane is counted and logged. See the control-plane audit section below.

## known security considerations

1. **No built-in auth.** The HTTP API trusts every caller. A reverse proxy or service mesh must enforce identity, and it must also authorize by path. Orla applies no per-path authorization of its own, so authenticating the caller does not stop an agent that reached `/v1/*` from also reaching `/api/v1/*` and rewriting the mapping every other agent runs on. Restrict the control plane to platform-engineer callers separately from the data plane.
2. **API key fan-out.** Orla holds outbound credentials for every backend. Compromise of the orla process exposes all of them.
3. **Rate limiting is per-instance.** `rate_per_second` is enforced per orla process. Multiple replicas multiply the effective cap.
4. **Tenant isolation is advisory.** Tenancy is carried by request headers such as `X-Orla-Tag-Tenant` and used for fair-share scheduling. It is not a security boundary on its own.

## control-plane audit

Orla reports every write to the control plane and serves all of them. Reporting is what makes the missing authorization boundary visible to an operator. Every request to `/api/v1/*` other than a read is counted in `orla_control_plane_mutations_total{resource, method, outcome}`, where `outcome` separates a write that changed state from one the API rejected.

The request log answers who. Every request line carries `remote_addr` and `forwarded_for` alongside the method, path, and status. Orla authenticates nobody, so both addresses are whatever the caller claimed. Use the pair to correlate a write with a source you already trust, and alert on control-plane writes arriving from anywhere but your platform-engineer path.

## JSON on every write

Orla answers 415 to a POST, PUT, or PATCH whose `Content-Type` is anything but `application/json`. GET, HEAD, and DELETE carry no body and stay unchecked.

The rule closes a path from the browser. A page can send `text/plain`, `application/x-www-form-urlencoded`, or `multipart/form-data` to another origin without asking permission first, so a page open in a developer's browser could otherwise post a JSON body to an orla running on that developer's machine and rewrite the mapping every agent runs on. Requiring `application/json` makes the browser ask permission first, and orla serves no preflight, so the write never leaves the page.

A client that already sends `application/json` is unaffected. `orlactl` and the OpenAI SDKs set the header on every request.

Thank you for helping keep orla secure!
