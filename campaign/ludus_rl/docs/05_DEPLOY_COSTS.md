# 05 · Deployment & costs — Vast.ai prototype → Hetzner → AWS

## Principle
GPU boxes are ephemeral and stateless; the control plane is small, stable, and cheap.
Workers PULL jobs from queues (survive box churn). Terraform/IaC from day 1 so moving
cloud = re-pointing, not rewriting. Billing alarms BEFORE scaling anything.

## Phase A — prototype/dev (now → P1): ~$40–70/mo typical
| Component | Where | $/mo |
|---|---|---|
| Control plane (API, Postgres, Redis, web) | Hetzner CX32-class VPS (or Contabo) | 8–15 |
| Replays/object storage | Cloudflare R2 (free egress) | 0–5 |
| GPU dev/training worker | Vast.ai 3090/4090 interruptible $0.15–0.35/hr, ON-DEMAND (stop when idle) | 25–45 (≈4h/day) |
| Judging (CPU jit match runner) | same VPS until >10k matches/day | 0 |
| DNS/TLS/CDN | Cloudflare free | ~1 |
Worst-case heavy month (24/7 GPU): $150–300. Networking: Tailscale VPS↔workers.
NEVER: databases or replays on the Vast box; static IP assumptions; push-based job dispatch.

## Phase B — early production (P2+): ~$100–200/mo
Hetzner dedicated (AX42-class, 8c/64GB, ~€46) runs API+DB+judging for thousands of users;
R2 for storage; GPU burst on Vast/RunPod. 5–10× cheaper than AWS-equivalent.
Move to managed Postgres (or nightly tested backups + WAL) before real users.

## Phase C — AWS (only when revenue / SLA / compliance demands): floor $250–550/mo
| Tier | Setup | $/mo |
|---|---|---|
| Control plane | ECS Fargate (2 svc) + ALB + RDS Postgres t4g→r6g + ElastiCache + S3+CloudFront | 180–320 |
| GPU judging | g6.xlarge SPOT ($0.35–0.45/hr), scale-to-zero on queue depth | 50–200 usage |
| Training rental | keep on Vast/RunPod even in prod (3–5× cheaper than AWS GPU) — resell with margin | net-positive |
| Observability | CloudWatch basics + Grafana Cloud free tier | 20–40 |

## Best practices checklist
- [ ] Everything behind a queue; workers scale-to-zero.
- [ ] Spot + checkpointing for ALL GPU workloads.
- [ ] S3/R2 lifecycle: hot replays 90 days → cold storage.
- [ ] Savings Plans only after 3 months of stable baseline.
- [ ] Budget alarms at $100/$300 (+ per-worker runtime caps — rogue-GPU protection).
- [ ] Secrets in SSM/vault, never in repo or images.
- [ ] One `terraform apply` from empty account to running stack.
