import dotenv from "dotenv";
import path from "node:path";
import fs from "node:fs";
import crypto from "node:crypto";
import { Document } from "@langchain/core/documents";
import { TokenTextSplitter } from "@langchain/textsplitters";
import { PineconeEmbeddings, PineconeStore } from "@langchain/pinecone";
import { Pinecone } from "@pinecone-database/pinecone";

const JIKAN_BASE_URL = "https://api.jikan.moe/v4";
const JIKAN_MIN_INTERVAL_MS = 1_050;
const DEFAULT_LIMIT = 25;
const DEFAULT_MAX_CHARACTERS = 12;
const DEFAULT_NAMESPACE = "jikan-mal";
const DEFAULT_CLOUD = "aws";
const DEFAULT_REGION = "us-east-1";
const DEFAULT_CHUNK_SIZE = 500;
const DEFAULT_CHUNK_OVERLAP = 50;
const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIMENSION = 1_536;

const TAG_RE = /<[^>]+>/g;
const BOILERPLATE_RE =
  /\[(?:Written|Adapted)\s+by\s+MAL\s+Rewrite\]|\bSource:\s*[^.\n]+/gi;

interface JikanPagination {
  has_next_page: boolean;
}

interface JikanTopResponse {
  pagination?: JikanPagination;
  data?: Array<{
    mal_id?: number;
  }>;
}

interface JikanFullResponse {
  data?: {
    mal_id?: number;
    title?: string | null;
    title_english?: string | null;
    title_japanese?: string | null;
    synopsis?: string | null;
    genres?: Array<{ name?: string | null }>;
  };
}

interface JikanCharactersResponse {
  data?: Array<{
    character?: {
      name?: string | null;
    };
  }>;
}

interface CliOptions {
  indexName: string;
  namespace: string;
  cloud: string;
  region: string;
  limitPerPage: number;
  maxPagesAnime: number;
  maxPagesManga: number;
  maxCharactersPerTitle: number;
}

class JikanRateLimitedClient {
  private nextRequestTs = 0;

  private async throttle(): Promise<void> {
    const now = Date.now();
    const waitMs = this.nextRequestTs - now;
    if (waitMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
    this.nextRequestTs = Date.now() + JIKAN_MIN_INTERVAL_MS;
  }

  async getJson<T>(resourcePath: string, attempts = 5): Promise<T> {
    const normalizedPath = resourcePath.startsWith("/") ? resourcePath : `/${resourcePath}`;

    for (let attempt = 0; attempt < attempts; attempt += 1) {
      await this.throttle();
      const response = await fetch(`${JIKAN_BASE_URL}${normalizedPath}`, {
        headers: {
          Accept: "application/json",
          "User-Agent": "PanelFlow-RAG-Ingestion/1.0"
        }
      });

      if (response.ok) {
        return (await response.json()) as T;
      }

      if ((response.status === 429 || response.status >= 500) && attempt < attempts - 1) {
        const retryAfterSeconds = Number.parseInt(response.headers.get("retry-after") ?? "0", 10);
        const delayMs = Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0
          ? retryAfterSeconds * 1_000
          : 1_250 * (attempt + 1);
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        continue;
      }

      const body = await response.text().catch(() => "");
      throw new Error(`Jikan request failed (${response.status}) ${normalizedPath} ${body.slice(0, 300)}`);
    }

    throw new Error(`Jikan request failed after retries: ${normalizedPath}`);
  }
}

function loadEnvironment(): void {
  const cwd = process.cwd();
  const candidates = [
    path.resolve(cwd, ".env"),
    path.resolve(cwd, "..", ".env"),
    path.resolve(cwd, "..", "..", ".env"),
    path.resolve(cwd, "..", "..", "..", ".env")
  ];
  for (const envPath of candidates) {
    if (fs.existsSync(envPath)) {
      dotenv.config({ path: envPath, override: false });
    }
  }
}

function decodeHtmlEntities(value: string): string {
  return value
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", "\"")
    .replaceAll("&#39;", "'")
    .replaceAll("&nbsp;", " ");
}

function cleanRawText(value: string | null | undefined): string {
  const decoded = decodeHtmlEntities(value ?? "");
  return decoded
    .replace(TAG_RE, " ")
    .replace(BOILERPLATE_RE, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getArgValue(args: string[], name: string): string | undefined {
  const prefix = `--${name}=`;
  for (const arg of args) {
    if (arg.startsWith(prefix)) {
      return arg.slice(prefix.length);
    }
  }
  return undefined;
}

function parseIntArg(args: string[], name: string, fallback: number): number {
  const raw = getArgValue(args, name);
  if (!raw) {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`Invalid --${name} value: ${raw}`);
  }
  return parsed;
}

function buildTimestampedIndexName(): string {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
  return `jikan-rag-${stamp}`.toLowerCase();
}

function parseCliOptions(argv: string[]): CliOptions {
  const args = argv.slice(2);
  const indexName = (getArgValue(args, "index-name") ?? buildTimestampedIndexName()).toLowerCase();
  const namespace = getArgValue(args, "namespace") ?? DEFAULT_NAMESPACE;
  const cloud = getArgValue(args, "cloud") ?? DEFAULT_CLOUD;
  const region = getArgValue(args, "region") ?? DEFAULT_REGION;

  return {
    indexName,
    namespace,
    cloud,
    region,
    limitPerPage: parseIntArg(args, "limit-per-page", DEFAULT_LIMIT),
    maxPagesAnime: parseIntArg(args, "max-pages-anime", 0),
    maxPagesManga: parseIntArg(args, "max-pages-manga", 0),
    maxCharactersPerTitle: parseIntArg(args, "max-characters", DEFAULT_MAX_CHARACTERS)
  };
}

async function ensurePineconeIndex(options: CliOptions, pinecone: Pinecone): Promise<void> {
  const indexes = await pinecone.listIndexes();
  const alreadyExists = (indexes.indexes ?? []).some((entry) => entry.name === options.indexName);
  if (alreadyExists) {
    console.log(`[index] using existing index: ${options.indexName}`);
    return;
  }

  console.log(`[index] creating ${options.indexName} (${options.cloud}/${options.region})`);
  await pinecone.createIndex({
    name: options.indexName,
    dimension: EMBEDDING_DIMENSION,
    metric: "cosine",
    spec: {
      serverless: {
        cloud: options.cloud as "aws" | "gcp" | "azure",
        region: options.region
      }
    },
    waitUntilReady: true,
    suppressConflicts: true
  });
}

function buildDocument(
  mediaType: "anime" | "manga",
  fullPayload: JikanFullResponse,
  charactersPayload: JikanCharactersResponse,
  maxCharactersPerTitle: number
): Document | null {
  const item = fullPayload.data;
  if (!item || typeof item.mal_id !== "number") {
    return null;
  }

  const title = cleanRawText(item.title_english || item.title || item.title_japanese || "");
  if (!title) {
    return null;
  }

  const genres = (item.genres ?? [])
    .map((genre) => cleanRawText(genre.name))
    .filter((genre) => genre.length > 0);

  const characters = (charactersPayload.data ?? [])
    .map((entry) => cleanRawText(entry.character?.name))
    .filter((name) => name.length > 0)
    .slice(0, maxCharactersPerTitle);

  const synopsis = cleanRawText(item.synopsis) || "No synopsis available.";
  const content = [
    `Title: ${title}`,
    `Genres: ${genres.length > 0 ? genres.join(", ") : "Unknown"}`,
    `Synopsis: ${synopsis}`,
    `Characters: ${characters.length > 0 ? characters.join(", ") : "Unknown"}`
  ].join("\n");

  return new Document({
    pageContent: content,
    metadata: {
      mal_id: item.mal_id,
      title,
      type: mediaType
    }
  });
}

function buildChunkId(document: Document, chunkIndex: number): string {
  const type = String(document.metadata.type ?? "unknown").toLowerCase();
  const malId = String(document.metadata.mal_id ?? "na").toLowerCase();
  const digest = crypto.createHash("sha1").update(document.pageContent).digest("hex").slice(0, 16);
  return `${type}-${malId}-${chunkIndex}-${digest}`;
}

async function processMediaType(options: {
  mediaType: "anime" | "manga";
  maxPages: number;
  limitPerPage: number;
  maxCharactersPerTitle: number;
  client: JikanRateLimitedClient;
  splitter: TokenTextSplitter;
  vectorStore: PineconeStore;
}): Promise<{ documents: number; chunks: number; pages: number }> {
  let page = 1;
  let pagesProcessed = 0;
  let totalDocs = 0;
  let totalChunks = 0;
  const seenIds = new Set<number>();

  while (true) {
    if (options.maxPages > 0 && pagesProcessed >= options.maxPages) {
      break;
    }

    const pagePayload = await options.client.getJson<JikanTopResponse>(
      `/top/${options.mediaType}?page=${page}&limit=${options.limitPerPage}`
    );

    const pageIds = (pagePayload.data ?? [])
      .map((item) => item.mal_id)
      .filter((id): id is number => Number.isFinite(id))
      .filter((id) => !seenIds.has(id));

    pageIds.forEach((id) => seenIds.add(id));
    pagesProcessed += 1;
    console.log(`[jikan:${options.mediaType}] page=${page} ids=${pageIds.length}`);

    const docs: Document[] = [];
    for (const id of pageIds) {
      try {
        const [full, chars] = await Promise.all([
          options.client.getJson<JikanFullResponse>(`/${options.mediaType}/${id}/full`),
          options.client.getJson<JikanCharactersResponse>(`/${options.mediaType}/${id}/characters`)
        ]);

        const doc = buildDocument(
          options.mediaType,
          full,
          chars,
          options.maxCharactersPerTitle
        );
        if (doc) {
          docs.push(doc);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[jikan:${options.mediaType}] skip mal_id=${id} reason=${message}`);
      }
    }

    if (docs.length > 0) {
      const chunks = await options.splitter.splitDocuments(docs);
      if (chunks.length > 0) {
        const ids = chunks.map((chunk, idx) => buildChunkId(chunk, idx));
        await options.vectorStore.addDocuments(chunks, { ids });
        totalDocs += docs.length;
        totalChunks += chunks.length;
        console.log(
          `[pinecone:${options.mediaType}] upserted docs=${docs.length} chunks=${chunks.length} ` +
            `totals docs=${totalDocs} chunks=${totalChunks}`
        );
      }
    }

    if (!pagePayload.pagination?.has_next_page) {
      break;
    }

    page += 1;
  }

  return {
    documents: totalDocs,
    chunks: totalChunks,
    pages: pagesProcessed
  };
}

async function main(): Promise<void> {
  loadEnvironment();
  const cli = parseCliOptions(process.argv);

  const pineconeApiKey = process.env.PINECONE_API_KEY?.trim();
  if (!pineconeApiKey) {
    throw new Error("Missing PINECONE_API_KEY in environment.");
  }

  const pinecone = new Pinecone({ apiKey: pineconeApiKey });
  await ensurePineconeIndex(cli, pinecone);

  const pineconeIndex = pinecone.index(cli.indexName);
  const embeddings = new PineconeEmbeddings({
    apiKey: pineconeApiKey,
    model: EMBEDDING_MODEL
  });

  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
    namespace: cli.namespace,
    maxConcurrency: 4
  });

  const splitter = new TokenTextSplitter({
    chunkSize: DEFAULT_CHUNK_SIZE,
    chunkOverlap: DEFAULT_CHUNK_OVERLAP
  });

  console.log(
    `[config] index=${cli.indexName} namespace=${cli.namespace} limit=${cli.limitPerPage} ` +
      `maxPagesAnime=${cli.maxPagesAnime || "all"} maxPagesManga=${cli.maxPagesManga || "all"} ` +
      `maxCharacters=${cli.maxCharactersPerTitle}`
  );

  const client = new JikanRateLimitedClient();
  const animeSummary = await processMediaType({
    mediaType: "anime",
    maxPages: cli.maxPagesAnime,
    limitPerPage: cli.limitPerPage,
    maxCharactersPerTitle: cli.maxCharactersPerTitle,
    client,
    splitter,
    vectorStore
  });

  const mangaSummary = await processMediaType({
    mediaType: "manga",
    maxPages: cli.maxPagesManga,
    limitPerPage: cli.limitPerPage,
    maxCharactersPerTitle: cli.maxCharactersPerTitle,
    client,
    splitter,
    vectorStore
  });

  console.log(
    `[done] anime docs=${animeSummary.documents} chunks=${animeSummary.chunks} pages=${animeSummary.pages}; ` +
      `manga docs=${mangaSummary.documents} chunks=${mangaSummary.chunks} pages=${mangaSummary.pages}`
  );
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  console.error(message);
  process.exitCode = 1;
});
