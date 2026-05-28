from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

HF_JSON_URI = "hf://datasets/abatilo/myanimelist-embeddings/anime-embeddings.json"
HF_PARQUET_CANDIDATES = [
    "hf://datasets/abatilo/myanimelist-embeddings/data/train-00000-of-00001.parquet",
    "hf://datasets/abatilo/myanimelist-embeddings/default/train-00000-of-00001.parquet",
    "hf://datasets/abatilo/myanimelist-embeddings/train-00000-of-00001.parquet",
]

LOGGER = logging.getLogger("bootstrap")


@dataclass
class BootstrapConfig:
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int
    bm25_values_path: Path
    text_key: str
    create_index: bool
    max_rows: int | None


def parse_args() -> BootstrapConfig:
    parser = argparse.ArgumentParser(description="Seed Pinecone from HF MAL dataset.")
    parser.add_argument("--pinecone-index", default=os.getenv("PINECONE_INDEX", "manga-rag-hybrid"))
    parser.add_argument("--pinecone-namespace", default=os.getenv("PINECONE_NAMESPACE", "baseline"))
    parser.add_argument("--pinecone-cloud", default=os.getenv("PINECONE_CLOUD", "aws"))
    parser.add_argument("--pinecone-region", default=os.getenv("PINECONE_REGION", "us-east-1"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "500")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("UPSERT_BATCH_SIZE", "100")))
    parser.add_argument(
        "--bm25-values-path",
        default=os.getenv("BM25_VALUES_PATH", "rag/ingestion/bm25_values.json"),
    )
    parser.add_argument("--text-key", default=os.getenv("PINECONE_TEXT_KEY", "chunk_text"))
    parser.add_argument(
        "--create-index",
        dest="create_index",
        action="store_true",
        default=os.getenv("CREATE_PINECONE_INDEX", "true").strip().lower() == "true",
    )
    parser.add_argument("--no-create-index", dest="create_index", action="store_false")
    parser.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()
    return BootstrapConfig(
        pinecone_index=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
        pinecone_cloud=args.pinecone_cloud,
        pinecone_region=args.pinecone_region,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        bm25_values_path=Path(args.bm25_values_path),
        text_key=args.text_key,
        create_index=args.create_index,
        max_rows=args.max_rows,
    )


def load_seed_dataframe() -> pd.DataFrame:
    LOGGER.info("Loading HF JSON dataset: %s", HF_JSON_URI)
    try:
        return pd.read_json(HF_JSON_URI)
    except Exception as json_exc:  # noqa: BLE001
        LOGGER.warning("JSON load failed (%s). Falling back to Parquet split(s).", json_exc)

    parquet_errors: list[str] = []
    for parquet_uri in HF_PARQUET_CANDIDATES:
        try:
            LOGGER.info("Trying HF Parquet: %s", parquet_uri)
            return pd.read_parquet(parquet_uri)
        except Exception as parquet_exc:  # noqa: BLE001
            parquet_errors.append(f"{parquet_uri}: {parquet_exc}")

    try:
        fs = HfFileSystem()
        parquet_files = fs.glob("datasets/abatilo/myanimelist-embeddings/**/*.parquet")
        if parquet_files:
            discovered_uri = f"hf://{parquet_files[0]}"
            LOGGER.info("Trying discovered HF Parquet: %s", discovered_uri)
            return pd.read_parquet(discovered_uri)
    except Exception as discover_exc:  # noqa: BLE001
        parquet_errors.append(f"dynamic_discovery: {discover_exc}")

    joined = "\n".join(parquet_errors)
    raise RuntimeError(f"Unable to load HF dataset from JSON or Parquet.\n{joined}")


def prepare_dataframe(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    frame = df.copy()

    if "embedding" in frame.columns:
        frame = frame.drop(columns=["embedding"])

    required = ["id", "title", "synopsis"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    frame = frame[required].copy()
    frame["id"] = pd.to_numeric(frame["id"], errors="coerce").astype("Int64")
    frame["title"] = frame["title"].fillna("").astype(str).str.strip()
    frame["synopsis"] = frame["synopsis"].fillna("").astype(str).str.strip()
    frame = frame.dropna(subset=["id"])
    frame = frame[(frame["title"] != "") & (frame["synopsis"] != "")]
    frame["id"] = frame["id"].astype(int)
    if isinstance(max_rows, int) and max_rows > 0:
        frame = frame.head(max_rows).copy()
        LOGGER.info("Applying max_rows cap: %d", max_rows)

    LOGGER.info("Prepared %d rows for ingestion.", len(frame))
    return frame


def to_documents(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    for row in df.itertuples(index=False):
        mal_id = int(row.id)
        title = str(row.title).strip()
        synopsis = str(row.synopsis).strip()

        # Required citation metadata schema.
        metadata = {"mal_id": mal_id, "title": title}
        text = f"Title: {title}\nSynopsis: {synopsis}"
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def chunk_documents(documents: list[Document], config: BootstrapConfig) -> list[Document]:
    splitter = TokenTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = splitter.split_documents(documents)
    LOGGER.info("Chunked %d documents into %d chunks.", len(documents), len(chunks))
    return chunks


def build_dense_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def fit_sparse_encoder(chunks: list[Document], bm25_values_path: Path) -> BM25Encoder:
    texts = [chunk.page_content for chunk in chunks]
    encoder = BM25Encoder().fit(texts)
    bm25_values_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(bm25_values_path))
    LOGGER.info("BM25 params saved to %s", bm25_values_path)
    return encoder


T = TypeVar("T")


def iter_batches(values: list[T], batch_size: int) -> Iterable[list[T]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def ensure_index(pc: Pinecone, config: BootstrapConfig, dense_dim: int) -> None:
    if pc.has_index(name=config.pinecone_index):
        return

    if not config.create_index:
        raise RuntimeError(f"Index '{config.pinecone_index}' does not exist and --create-index is false.")

    LOGGER.info("Creating Pinecone index %s (dim=%d)", config.pinecone_index, dense_dim)
    pc.create_index(
        name=config.pinecone_index,
        dimension=dense_dim,
        metric="dotproduct",
        spec=ServerlessSpec(cloud=config.pinecone_cloud, region=config.pinecone_region),
    )


def upsert_chunks(
    *,
    pc: Pinecone,
    config: BootstrapConfig,
    chunks: list[Document],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[str, list[float] | list[int]]],
) -> None:
    if not chunks:
        LOGGER.warning("No chunks to upsert.")
        return

    ensure_index(pc, config, dense_dim=len(dense_vectors[0]))
    index = pc.Index(config.pinecone_index)

    records = []
    for idx, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vectors, sparse_vectors, strict=True)):
        mal_id = int(chunk.metadata["mal_id"])
        title = str(chunk.metadata["title"])

        # Keep citation keys exact and explicit. `chunk_text` is included for retrieval rendering.
        metadata = {"mal_id": mal_id, "title": title, config.text_key: chunk.page_content}
        record = {
            "id": f"mal-{mal_id}-chunk-{idx:06d}",
            "values": dense,
            "sparse_values": sparse,
            "metadata": metadata,
        }
        records.append(record)

    for batch in iter_batches(records, config.batch_size):
        index.upsert(vectors=batch, namespace=config.pinecone_namespace)
    LOGGER.info("Upserted %d vectors into %s/%s.", len(records), config.pinecone_index, config.pinecone_namespace)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    load_dotenv()
    config = parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is required.")

    frame = prepare_dataframe(load_seed_dataframe(), max_rows=config.max_rows)
    documents = to_documents(frame)
    chunks = chunk_documents(documents, config)

    embeddings = build_dense_embeddings()
    LOGGER.info("Embedding %d chunks with BAAI/bge-small-en-v1.5", len(chunks))
    dense_vectors = embeddings.embed_documents([doc.page_content for doc in chunks])

    bm25 = fit_sparse_encoder(chunks, config.bm25_values_path)
    sparse_vectors = bm25.encode_documents([doc.page_content for doc in chunks])

    pc = Pinecone(api_key=api_key)
    upsert_chunks(
        pc=pc,
        config=config,
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
    )


if __name__ == "__main__":
    main()
