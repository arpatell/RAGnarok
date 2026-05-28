export type IngestErrorCode =
  | "INVALID_URL"
  | "UNSUPPORTED_DOMAIN"
  | "NOT_PUBLIC"
  | "NOT_CHAPTER"
  | "INGEST_FAILED";

export class IngestError extends Error {
  public readonly code: IngestErrorCode;
  public readonly status: number;

  constructor(code: IngestErrorCode, message: string, status = 400) {
    super(message);
    this.name = "IngestError";
    this.code = code;
    this.status = status;
  }
}

export function isIngestError(value: unknown): value is IngestError {
  return value instanceof IngestError;
}

export function toErrorPayload(error: IngestError): {
  error: string;
  code: IngestErrorCode;
} {
  return {
    error: error.message,
    code: error.code
  };
}
