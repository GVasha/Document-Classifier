export type DocLabel = "email" | "invoice" | "questionnaire" | "scientific_pub";

export interface InvoiceFields {
  invoice_number: string | null;
  invoice_date: string | null;
  due_date: string | null;
  issuer_name: string | null;
  recipient_name: string | null;
  total_amount: string | null;
}

export interface ClassifyResponse {
  label: DocLabel;
  invoice_fields?: InvoiceFields;
}

export interface HealthResponse {
  status: string;
  models_present: boolean;
}
