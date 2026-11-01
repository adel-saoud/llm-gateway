// Open-loop load against the gateway: requests start at a fixed arrival rate
// whatever the response time, so a slow gateway is measured under the load it
// was asked to take (no coordinated omission from a closed client loop).
//
// Driven by scripts/bench_overhead.py --open-loop, which runs it in the
// grafana/k6 Docker image. Inputs come from environment variables.

import http from "k6/http";
import { Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL;
const API_KEY = __ENV.API_KEY;
const RATE = Number(__ENV.RATE || "200");
const DURATION = __ENV.DURATION || "30s";
const MODEL = __ENV.MODEL;

// Gateway self-time, parsed from the Server-Timing header.
const gatewayOverhead = new Trend("gateway_overhead_ms", true);

export const options = {
  scenarios: {
    open_loop: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1s",
      duration: DURATION,
      preAllocatedVUs: Math.max(10, Math.ceil(RATE / 10)),
      maxVUs: Math.max(50, RATE),
    },
  },
  summaryTrendStats: ["p(50)", "p(95)", "p(99)", "max"],
};

const body = JSON.stringify({
  model: MODEL,
  messages: [{ role: "user", content: "benchmark prompt" }],
});
const params = {
  headers: { Authorization: `Bearer ${API_KEY}`, "Content-Type": "application/json" },
};

export default function () {
  const res = http.post(`${BASE_URL}/v1/chat/completions`, body, params);
  const timing = res.headers["Server-Timing"] || "";
  const match = timing.match(/gateway;dur=([0-9.]+)/);
  if (match) {
    gatewayOverhead.add(Number(match[1]));
  }
}

export function handleSummary(data) {
  const pick = (name) => (data.metrics[name] ? data.metrics[name].values : null);
  return {
    stdout: JSON.stringify(
      {
        rate_per_s: RATE,
        duration: DURATION,
        http_req_duration_ms: pick("http_req_duration"),
        gateway_overhead_ms: pick("gateway_overhead_ms"),
        http_req_failed: pick("http_req_failed"),
        dropped_iterations: pick("dropped_iterations"),
        iterations: pick("iterations"),
      },
      null,
      2,
    ),
  };
}
