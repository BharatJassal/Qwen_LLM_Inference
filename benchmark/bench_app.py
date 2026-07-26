import asyncio
import time
import statistics
import httpx

URL = "http://localhost:8000/generate"
PAYLOAD = {"prompt": "Write me a story about ", "max_tokens": 100}
CONCURRENCY_LEVELS = [1, 4, 8]


async def one_request(client):
    """Send one request, return its latency in seconds."""
    start = time.time()
    resp = await client.post(URL, json=PAYLOAD, timeout=120.0)
    elapsed = time.time() - start
    assert resp.status_code == 200
    return elapsed


async def run_level(n):
    """Fire n requests simultaneously, return list of latencies."""
    async with httpx.AsyncClient() as client:
        tasks = []
        for _ in range(n):
            tasks.append(one_request(client))
        wall_start = time.time()
        latencies = await asyncio.gather(*tasks)
        wall_time = time.time() - wall_start
    return latencies, wall_time


def percentile(values, p):
    values = sorted(values)
    k = int(round((p / 100) * (len(values) - 1)))
    return values[k]


async def main():
    print(f"{'clients':>8} | {'p50 (s)':>8} | {'p95 (s)':>8} | {'wall (s)':>8} | {'req/s':>7}")
    for n in CONCURRENCY_LEVELS:
        latencies, wall_time = await run_level(n)
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        throughput = n / wall_time
        print(f"{n:>8} | {p50:>8.2f} | {p95:>8.2f} | {wall_time:>8.2f} | {throughput:>7.2f}")


if __name__ == "__main__":
    asyncio.run(main())