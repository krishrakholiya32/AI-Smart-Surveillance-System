#!/bin/sh
# Bootstraps nginx/certs/ with a temporary self-signed certificate so nginx
# can start listening on 443 before a real one exists. Safe to re-run —
# it's a no-op once real files (e.g. from Let's Encrypt) are in place.
#
# Usage: scripts/init_certs.sh
set -e

CERT_DIR="$(dirname "$0")/../nginx/certs"
mkdir -p "$CERT_DIR"

if [ -f "$CERT_DIR/fullchain.pem" ] && [ -f "$CERT_DIR/privkey.pem" ]; then
    echo "Certs already present in $CERT_DIR — leaving them as-is."
    exit 0
fi

echo "No certs found — generating a temporary self-signed certificate..."
openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout "$CERT_DIR/privkey.pem" \
    -out "$CERT_DIR/fullchain.pem" \
    -subj "/CN=localhost"

echo "Done. nginx can now start on :443 with this self-signed cert."
echo "Browsers will show a 'not trusted' warning until you replace it with"
echo "a real Let's Encrypt certificate — see DEPLOY.md."
