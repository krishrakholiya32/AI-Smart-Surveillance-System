# Deployment Guide — Oracle Cloud Free Tier

## Why Oracle Cloud Always Free?

| Resource | Free Amount |
|---|---|
| Arm A1 VM (ampere) | **4 OCPUs + 24 GB RAM** — never expires |
| Block Storage | 200 GB |
| Object Storage | 20 GB |
| Outbound data | 10 TB/month |

This is orders of magnitude more than AWS/GCP free tiers and is **permanent** (not a trial).

---

## Step 1 — Create Oracle Cloud Account

1. Go to [cloud.oracle.com](https://cloud.oracle.com) → **Start for free**
2. Use your email — a credit card is required for identity but **will not be charged** if you stay within Always Free limits
3. Choose your **Home Region** (closest to you, cannot be changed later)

---

## Step 2 — Create the ARM VM

1. In OCI Console → **Compute → Instances → Create Instance**
2. **Name**: `surveillance-server`
3. **Image**: Ubuntu 22.04 (Always Free eligible)
4. **Shape**: Change → **Ampere → VM.Standard.A1.Flex**
   - OCPUs: `4` (max free)
   - Memory: `24 GB` (max free)
5. **SSH Keys**: Upload your public key (`~/.ssh/id_rsa.pub`)
   - If you don't have one: `ssh-keygen -t rsa -b 4096`
6. **Boot Volume**: 100 GB (free up to 200 GB total)
7. Click **Create**

Wait ~2 minutes. Note the **Public IP address**.

---

## Step 3 — Configure Security (Open ports)

OCI blocks all ports by default.

1. Compute → Instances → Your instance → **Subnet** link → **Security List**
2. **Ingress Rules** → Add:
   | Protocol | Port | Source CIDR | Description |
   |---|---|---|---|
   | TCP | 22 | Your home/known IP only (e.g. `203.0.113.5/32`) | SSH |
   | TCP | 80 | `0.0.0.0/0` | HTTP — needed publicly for the Let's Encrypt challenge, and to redirect to HTTPS |
   | TCP | 443 | `0.0.0.0/0` | HTTPS — the actual app, once you've pointed a domain at this VM |

   Avoid `0.0.0.0/0` for SSH. If your IP changes, update the rule rather than opening it to everyone.
3. Also open the OS firewall on the VM (matching the same restriction):
```bash
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

---

## Step 4 — SSH into the VM & Install Docker

```bash
ssh ubuntu@YOUR_VM_IP
```

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose plugin
sudo apt install -y docker-compose-plugin

# Verify
docker --version
docker compose version
```

---

## Step 5 — Transfer Project Files

**On your local machine:**

```bash
# Option A: Clone from GitHub (recommended)
# Push your code to GitHub first, then on the VM:
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git ~/surveillance

# Option B: SCP from local machine
scp -r "AI Smart Surveillance System/" ubuntu@YOUR_VM_IP:~/surveillance
```

---

## Step 6 — Upload YOLO Model Files

The model files (`.pt`) are large and not in git.
Transfer them to the `models/` folder on the server:

```bash
# From your local machine:
scp yolo11s.pt yolo11n-pose.pt weapon.pt ubuntu@YOUR_VM_IP:~/surveillance/models/
```

That's it — the backend automatically exports these to ONNX (2-3x faster CPU inference) the first time the container starts, and reuses the exported files on every restart. No manual export step needed.

---

## Step 7 — Configure Environment

```bash
cd ~/surveillance
cp .env.example .env
nano .env
```

Fill in:
```env
SECRET_KEY=<paste output of: openssl rand -hex 32>
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<your strong password>
SERVER_IP=<your VM public IP>
```

---

## Step 8 — Bootstrap a Certificate, Then Build and Launch

nginx listens on 443 unconditionally, so it needs *some* certificate file
present before it'll start — even a temporary one:

```bash
cd ~/surveillance
bash scripts/init_certs.sh    # generates a self-signed cert into nginx/certs/
```

Then build and launch as normal:

```bash
docker compose build
docker compose up -d

# Verify all services are running
docker compose ps
docker compose logs -f backend   # watch for startup errors
```

Expected output from `docker compose ps`:
```
NAME          STATUS    PORTS
db            running   5432/tcp
backend       running   0.0.0.0:8000->8000/tcp
frontend      running   0.0.0.0:3000->80/tcp
nginx         running   0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
```

First boot will take 1-2 minutes longer than usual — the backend exports the YOLO models to ONNX once before starting. Watch `docker compose logs -f backend` to see progress.

---

## Step 9 — Access the System

Open your browser: **https://YOUR_VM_IP** (you'll get a "not trusted" warning
from the self-signed bootstrap cert — that's expected until Step 10).
Plain `http://YOUR_VM_IP` still works too at this point.

Login with:
- Username: `admin`
- Password: whatever you set in `.env`

### First steps in the UI:
1. Go to **Cameras** → Add your first camera
   - For a DroidCam phone: source = `http://PHONE_IP:4747/video`
   - For an RTSP IP camera: source = `rtsp://user:pass@CAMERA_IP/stream`
2. Start the camera
3. Go to **Camera View** → **Zone Setup** tab → draw restricted zones (edits apply live, no restart needed)
4. Watch the **Dashboard** for live feeds and alerts

---

## Step 10 — Switch to a Real Let's Encrypt Certificate

So far you're running on the self-signed cert from `scripts/init_certs.sh` —
browsers will show a "not trusted" warning at `https://YOUR_VM_IP`. Once you
have a domain, replace it with a real one:

1. **Point DNS at the VM.** Create an A record for your domain (e.g.
   `surveillance.example.com`) pointing at the VM's public IP. Wait for it
   to propagate (`dig +short surveillance.example.com` should return the IP).

2. **Confirm port 80 is reachable from the internet** (Step 3) — Let's
   Encrypt's HTTP-01 challenge needs it, even though the app itself will
   end up served on 443.

3. **Issue the certificate** using the `certbot` service already defined in
   `docker-compose.yml` (it doesn't start automatically — `--profile certbot`
   opts it in for this one-off run):
   ```bash
   docker compose --profile certbot run --rm certbot certonly \
     --webroot --webroot-path=/var/www/certbot \
     -d surveillance.example.com \
     --email you@example.com --agree-tos --no-eff-email
   ```
   This writes `live/surveillance.example.com/fullchain.pem` and `privkey.pem`
   under `nginx/certs/` (the container's `/etc/letsencrypt`).

4. **Point nginx at the real cert.** Either symlink the issued files over the
   self-signed ones, or edit the two `ssl_certificate*` lines in
   `nginx/nginx.conf` to the `live/surveillance.example.com/` paths:
   ```bash
   ln -sf live/surveillance.example.com/fullchain.pem nginx/certs/fullchain.pem
   ln -sf live/surveillance.example.com/privkey.pem   nginx/certs/privkey.pem
   docker compose restart nginx
   ```

5. **Force HTTPS.** Uncomment the `return 301 https://$host$request_uri;`
   line in the `:80` server block of `nginx/nginx.conf`, then
   `docker compose restart nginx`.

6. **Set up auto-renewal.** Let's Encrypt certs expire every 90 days. Add a
   host cron job:
   ```bash
   (crontab -l 2>/dev/null; echo "0 3 * * * cd ~/surveillance && docker compose --profile certbot run --rm certbot renew && docker compose restart nginx") | crontab -
   ```

After this, access the app at `https://surveillance.example.com` with no
browser warning, and `http://` requests auto-redirect to it.

---

## Step 11 — CI/CD with GitHub Actions (auto-deploy on push)

1. In your GitHub repo → **Settings → Secrets → Actions** → Add:
   - `SERVER_HOST`: your VM public IP
   - `SERVER_USER`: `ubuntu`
   - `SSH_PRIVATE_KEY`: contents of `~/.ssh/id_rsa` (your local private key)

2. Push to `main` branch — the `.github/workflows/deploy.yml` will automatically deploy.

---

## Maintenance

```bash
# View logs
docker compose logs -f backend

# Update and redeploy
git pull && docker compose build && docker compose up -d

# View alert snapshots
ls ~/surveillance/  # look in the alert_storage Docker volume

# Backup database
docker compose exec db pg_dump -U surveillance surveillance > backup.sql

# Monitor resource usage
docker stats
htop
```

---

## Performance on Oracle A1 (4 OCPU / 24 GB)

This deployment is sized for **one camera**. The pipeline runs 3 YOLO models per frame on CPU only (no GPU on the free tier), so per-camera throughput is the limiting factor, not raw core count.

| Mode | Expected FPS (single camera) |
|---|---|
| YOLOv11 `.pt` (PyTorch, CPU) | 5–10 FPS |
| YOLOv11 `.onnx` (ONNX Runtime, CPU) — used automatically | 10–18 FPS |

The backend's worker is paced to a 10 FPS target, so anything above that is headroom, not wasted capacity. Running a second camera will roughly halve both numbers — benchmark before relying on more than one.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Backend fails to start | Check `docker compose logs backend` — usually a missing `SECRET_KEY`/`ADMIN_PASSWORD` in `.env`, wrong `DATABASE_URL`, or missing model files |
| Can't reach port 80/443 | Check OCI Security List ingress rules AND OS iptables rules |
| nginx won't start at all | It needs cert files to exist before listening on 443 — run `bash scripts/init_certs.sh` first |
| Models not loading | Ensure `.pt` files are in `~/surveillance/models/` |
| Low FPS on cloud | Confirm `.onnx` files exist in `~/surveillance/models/` after first boot — if not, check backend logs for export errors |
| WebSocket won't connect | Ensure nginx is forwarding `/ws/` correctly; check browser console |
| Locked out of login | 5 failed attempts locks that username for 15 minutes (in-memory, resets on backend restart) — just wait it out |
| Certbot challenge fails | Confirm DNS has propagated (`dig +short your-domain.com`) and port 80 is reachable from the public internet, not just your IP |
