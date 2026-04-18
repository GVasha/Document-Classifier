import type { ClassifyResponse, HealthResponse } from "./types";

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json() as Promise<HealthResponse>;
}

export async function classifyFile(file: File): Promise<ClassifyResponse> {
  const body = new FormData();
  body.append("file", file, file.name);

  const res = await fetch("/api/classify", {
    method: "POST",
    body,
  });

  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const err = (await res.json()) as { detail?: string | { msg?: string }[] };
      if (typeof err.detail === "string") msg = err.detail;
      else if (Array.isArray(err.detail) && err.detail[0]?.msg) msg = err.detail[0].msg;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }

  return res.json() as Promise<ClassifyResponse>;
}
