import { useCallback, useEffect, useState } from "react";
import { classifyFile, fetchHealth } from "./api";
import type { ClassifyResponse, DocLabel, InvoiceFields } from "./types";
import "./App.css";

const LABEL_HELP: Record<DocLabel, string> = {
  email: "Thread-style correspondence with headers like From / To / Subject.",
  invoice: "Billing document; the pipeline may extract structured invoice fields.",
  questionnaire: "Forms or surveys with questions and response cues.",
  scientific_pub: "Academic paper signals such as abstract and references.",
};

function badgeClass(label: DocLabel): string {
  return `badge badge-${label}`;
}

function formatField(value: string | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  return value;
}

const INVOICE_KEYS: { key: keyof InvoiceFields; title: string }[] = [
  { key: "invoice_number", title: "Invoice number" },
  { key: "invoice_date", title: "Invoice date" },
  { key: "due_date", title: "Due date" },
  { key: "issuer_name", title: "Issuer" },
  { key: "recipient_name", title: "Recipient" },
  { key: "total_amount", title: "Total amount" },
];

export default function App() {
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => setHealthOk(h.models_present))
      .catch(() => setHealthOk(false));
  }, []);

  const onFiles = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    setFile(files[0]);
    setResult(null);
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      onFiles(e.dataTransfer.files);
    },
    [onFiles]
  );

  const runClassify = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await classifyFile(file);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }, [file]);

  return (
    <div className="shell">
      <header className="hero">
        <h1>Document Classifier</h1>
        <p>
          Upload a PDF or scanned image. The ensemble model labels it as email, invoice, questionnaire, or
          scientific publication — and extracts key fields when the document is an invoice.
        </p>
      </header>

      <div className="statusRow">
        <div className={`pill ${healthOk ? "pillOk" : healthOk === false ? "pillWarn" : ""}`}>
          <span className="pillDot" aria-hidden />
          {healthOk === null && "Checking API…"}
          {healthOk === true && "API & models ready"}
          {healthOk === false && "API unreachable or models missing"}
        </div>
      </div>

      <section
        className={`dropCard ${drag ? "drag" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <input
          className="fileInput"
          type="file"
          accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp"
          onChange={(e) => onFiles(e.target.files)}
        />
        <div className="dropIcon" aria-hidden>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="8" y1="13" x2="16" y2="13" />
            <line x1="8" y1="17" x2="14" y2="17" />
          </svg>
        </div>
        <h2 className="dropTitle">Drop a document here</h2>
        <p className="dropHint">PDF, PNG, JPG, TIFF, or BMP — max size depends on your server limits.</p>
        <div className="actions">
          <span className="btn btnPrimary" style={{ pointerEvents: "none" }}>
            Choose file
          </span>
        </div>
        {file && (
          <p className="fileName">
            Selected: <strong>{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
          </p>
        )}
      </section>

      <div className="actions" style={{ marginTop: "1.25rem" }}>
        <button type="button" className="btn btnPrimary" disabled={!file || loading} onClick={runClassify}>
          {loading ? "Classifying…" : "Run classification"}
        </button>
        <button
          type="button"
          className="btn btnGhost"
          disabled={loading}
          onClick={() => {
            setFile(null);
            setResult(null);
            setError(null);
          }}
        >
          Clear
        </button>
      </div>

      {loading && (
        <div className="loader" style={{ marginTop: "1.25rem", justifyContent: "center" }}>
          <span className="spinner" aria-hidden />
          OCR and models can take a few seconds…
        </div>
      )}

      {error && <div className="errorBox">{error}</div>}

      {result && (
        <section className="result" aria-live="polite">
          <div className="resultHeader">
            <h3 className="resultTitle">Prediction</h3>
            <span className={badgeClass(result.label)}>{result.label.replaceAll("_", " ")}</span>
          </div>
          <p style={{ margin: "0 0 1.25rem", color: "var(--text-muted)", fontSize: "0.95rem" }}>
            {LABEL_HELP[result.label]}
          </p>

          {result.label === "invoice" && result.invoice_fields && (
            <>
              <h4
                style={{
                  margin: "0 0 1rem",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: "var(--text-muted)",
                }}
              >
                Extracted invoice fields
              </h4>
              <div className="invoiceGrid">
                {INVOICE_KEYS.map(({ key, title }) => {
                  const raw = result.invoice_fields![key];
                  const empty = raw === null || raw === undefined || raw === "";
                  return (
                    <div key={key} className="fieldCard">
                      <p className="fieldLabel">{title}</p>
                      <p className={`fieldValue mono ${empty ? "empty" : ""}`}>{formatField(raw)}</p>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </section>
      )}

      <p className="footerNote">
        Start the Python API from the repo root:{" "}
        <code className="mono" style={{ color: "var(--text)" }}>
          uvicorn api_server:app --reload --port 8000
        </code>
        , then <code className="mono">npm run dev</code> in <span className="mono">frontend/</span>.
      </p>
    </div>
  );
}
