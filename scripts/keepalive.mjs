// Pings the Render backend every 180 seconds so the free-tier instance
// doesn't go to sleep. Run with:
//   BACKEND_URL=https://your-service.onrender.com node scripts/keepalive.mjs
// Requires Node 18+ (uses the built-in fetch).

const RAW_URL = process.env.BACKEND_URL;
const INTERVAL_MS = 180_000;
const REQUEST_TIMEOUT_MS = 60_000;

if (!RAW_URL) {
  console.error("Missing BACKEND_URL. Example: BACKEND_URL=https://pothole-api.onrender.com node scripts/keepalive.mjs");
  process.exit(1);
}

const HEALTH_URL = `${RAW_URL.replace(/\/$/, "")}/health`;

async function ping() {
  const startedAt = Date.now();
  const stamp = new Date().toISOString();
  try {
    const res = await fetch(HEALTH_URL, {
      method: "GET",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    const elapsed = Date.now() - startedAt;
    console.log(`[${stamp}] ${res.status} ${res.statusText} in ${elapsed}ms`);
  } catch (err) {
    const elapsed = Date.now() - startedAt;
    console.warn(`[${stamp}] ping failed after ${elapsed}ms: ${err.message}`);
  }
}

console.log(`Pinging ${HEALTH_URL} every ${INTERVAL_MS / 1000}s. Ctrl+C to stop.`);
await ping();
setInterval(ping, INTERVAL_MS);
