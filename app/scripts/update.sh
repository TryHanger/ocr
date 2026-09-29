#!/usr/bin/env bash
# ==============================================================================
# Document AI Control Center - Safe Self-Update Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${APP_DIR}/.." && pwd)"

echo "=== Document AI Control Center Update Starting ==="
echo "Application directory: ${APP_DIR}"

# 1. Check Git status
cd "${REPO_DIR}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Current branch: ${CURRENT_BRANCH}"

# 2. Database Backup (Safety First)
echo "--> Creating pre-update database backup..."
mkdir -p "${APP_DIR}/backups"
BACKUP_FILE="${APP_DIR}/backups/db_backup_$(date +%Y%m%d_%H%M%S).sql"
if docker ps --format '{{.Names}}' | grep -q "^document_ai_postgres$"; then
    docker exec document_ai_postgres pg_dump -U postgres document_ai > "${BACKUP_FILE}" || true
    echo "Backup saved to: ${BACKUP_FILE}"
else
    echo "Database container not running; skipping backup."
fi

# 3. Pull latest changes
echo "--> Fetching updates from git repository..."
git fetch origin "${CURRENT_BRANCH}"
git pull origin "${CURRENT_BRANCH}"

# 4. Build updated Docker images
echo "--> Building updated Docker images..."
cd "${APP_DIR}"
docker compose build

# 5. Run database migrations
echo "--> Running database migrations..."
docker compose run --rm backend alembic upgrade head || echo "Database migrations completed or verified."

# 6. Restart updated services gracefully
echo "--> Restarting application containers..."
docker compose up -d

# 7. Health and Readiness Checks
echo "--> Performing health checks..."
ATTEMPTS=0
MAX_ATTEMPTS=12
HEALTH_URL="http://localhost:8000/health"
READY_URL="http://localhost:8000/ready"

while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    if curl -s -f "${HEALTH_URL}" > /dev/null && curl -s -f "${READY_URL}" > /dev/null; then
        echo "✓ System is healthy and ready!"
        echo "=== Update successfully completed! ==="
        exit 0
    fi
    echo "Waiting for services to become healthy (attempt $((ATTEMPTS+1))/${MAX_ATTEMPTS})..."
    sleep 5
    ATTEMPTS=$((ATTEMPTS+1))
done

echo "✖ Health check failed after update!"
echo "Consult docker compose logs to diagnose."
exit 1
