# Deploying the L3 Harvester on AWS EC2

`harvest_bitfinex_l3.py` runs unattended for a few days collecting
Bitfinex's public `book(R0)` + `trades` channels for BTC/ETH/SOL. This
doc is the AWS-side setup. Net cost under free tier: ~$0; out-of-pocket
post-free-tier: ~$3/month.

## 1. EC2 instance

| Setting | Value | Notes |
|---|---|---|
| Instance type | `t3.micro` | Free tier 12 months; ~$8/mo after |
| AMI | Amazon Linux 2023 (or Ubuntu 22.04) | Either works |
| Storage | 30 GB EBS gp3 | ~$2.40/mo; multi-day 3-coin L3 ≈ ~1–5 GB compressed |
| Security group | Egress 443/TCP to `api-pub.bitfinex.com` | No inbound needed |
| Region | `eu-west-1` or `us-east-1` | Bitfinex infra primarily in EU; `eu-west-1` minimises WS latency |
| Key pair | (your existing) | Required for SSH |

Launch via the AWS console or `aws ec2 run-instances`. Note the
instance's public IP for SSH.

## 2. Initial setup (one-time, after SSH)

```bash
# Amazon Linux 2023 ships Python 3.9; install 3.11 (websockets needs it)
sudo dnf install -y python3.11 python3.11-pip tmux git

# Clone the repo
git clone https://github.com/mihirhere/Disruption_Engine_Alpaca.git
cd Disruption_Engine_Alpaca

# Install the harvester's only dependency
python3.11 -m pip install --user websockets

# Make the data directory
mkdir -p ~/l3_data
```

## 3. Launch the harvester

```bash
# Run inside tmux so it survives SSH disconnects
tmux new -d -s harvester "python3.11 research/path_h_l3/harvest_bitfinex_l3.py \
    --out-dir ~/l3_data \
    --log-file ~/l3_data/harvester.log \
    --status-interval-s 300"

# Verify it started
tmux ls
tail -f ~/l3_data/harvester.log
```

Expected first few log lines:

```
2026-05-12T19:00:00Z [INFO] writing to /home/ec2-user/l3_data/bitfinex_l3_20260512.jsonl.gz
2026-05-12T19:00:00Z [INFO] connecting to wss://api-pub.bitfinex.com/ws/2
2026-05-12T19:00:00Z [INFO] subscribed: 6 channels across symbols=['tBTCUSD', 'tETHUSD', 'tSOLUSD']
2026-05-12T19:00:00Z [INFO] ack: book tBTCUSD chanId=...
2026-05-12T19:00:00Z [INFO] all subscriptions confirmed
2026-05-12T19:05:00Z [INFO] status: 30000 events (~100/s), 2495 KB written in last 300s
```

If the status line shows >50 events/s across 3 symbols, the harvester is
collecting at the expected rate. If it shows 0, the subscription failed
silently — `tail` the log file for an `error` event from Bitfinex
(invalid symbol, server-side rate limit, etc.).

## 4. Monitoring (during the run)

```bash
# Re-attach to tmux (Ctrl+B then D to detach)
tmux attach -t harvester

# Or just watch from outside
tail -f ~/l3_data/harvester.log

# Disk usage — sanity-check growth rate
du -sh ~/l3_data/
ls -lh ~/l3_data/
```

Expect ~300–800 MB/day compressed across 3 symbols during normal flow,
higher during volatility. Daily file rotates automatically at UTC
midnight.

## 5. Stop and download

After 3–4 days of data:

```bash
# Stop the harvester cleanly (SIGINT triggers the finally block, gz close)
tmux send-keys -t harvester C-c
sleep 2
tmux kill-session -t harvester

# Verify all files closed (no .gz currently being written)
ls -lh ~/l3_data/

# Bundle for transfer
tar czf l3_data.tgz l3_data/

# Option A — direct scp down to your workstation
scp -i <your-key.pem> ec2-user@<instance-ip>:~/l3_data.tgz .

# Option B — via S3 (useful if download speed is bad over residential)
aws s3 cp l3_data.tgz s3://<your-bucket>/path-h-l3/
# then on your workstation:  aws s3 cp s3://<bucket>/path-h-l3/l3_data.tgz .

# Once downloaded locally, unpack and the aggregator can run on the gz files
tar xzf l3_data.tgz
python research/path_h_l3/aggregate_mbo_events.py --vendor bitfinex \
    --in l3_data/bitfinex_l3_20260514.jsonl.gz \
    --out calibration/l3_ticks_combined.csv \
    --tick-events 100
```

(Note: `aggregate_mbo_events.py` currently has parser stubs for
`databento` / `tardis` / `bitfinex` / `synthetic`. The `parse_bitfinex_l3`
function in particular is still pending wire-up — see the README in this
directory under "Outstanding work" for the field mapping.)

## 6. Cost expectations

| Item | Free tier | Out-of-pocket after |
|---|---|---|
| t3.micro instance-hours | 750 hr/mo free | ~$8/mo |
| 30 GB EBS gp3 | 30 GB free | ~$2.40/mo |
| Outbound data transfer | 100 GB/mo free | $0.09/GB |
| **Total for a 4-day run within free tier** | | **$0** |

After the 12-month free tier expires, ~$11/mo if left running 24/7;
shut the instance down immediately after the data is downloaded to
avoid further accrual.

## 7. Decision gate after capture

Per [L3_RESEARCH_PLAN.md §6](L3_RESEARCH_PLAN.md), the post-acquisition
flow is:

- Wire `parse_bitfinex_l3` in `aggregate_mbo_events.py`
- Run aggregation: raw `.jsonl.gz` → event-clock tick CSV
- Distributional sanity check on the L3-derived features
- Train TCN on the L3 feature stack (`training/train_stream.py`)
- Backtest with `training/backtest_directional.py`
- If gross edge per trade ≥ 2 bps (Phase 5 threshold), proceed to
  AlphaEngine integration; otherwise document negative result and
  consider further pivots.
