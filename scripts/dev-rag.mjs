import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const host = process.env.RAG_HOST?.trim() || "0.0.0.0";
const port = process.env.RAG_PORT?.trim() || "8090";
const enableReload = /^(1|true|yes)$/i.test(process.env.RAG_RELOAD?.trim() || "");

const uvicornModuleArgs = [
  "-m",
  "uvicorn",
  "rag.rag_api:app",
  "--host",
  host,
  "--port",
  port
];
if (enableReload) {
  uvicornModuleArgs.push("--reload");
}

const cwd = process.cwd();
const candidates = [];

if (process.platform === "win32") {
  const venvPython = path.resolve(cwd, ".venv", "Scripts", "python.exe");
  if (existsSync(venvPython)) {
    candidates.push({ command: venvPython, args: uvicornModuleArgs, label: ".venv python" });
  }
  candidates.push({ command: "python", args: uvicornModuleArgs, label: "python" });
  candidates.push({ command: "py", args: ["-3", ...uvicornModuleArgs], label: "py -3" });
} else {
  const venvPython = path.resolve(cwd, ".venv", "bin", "python");
  if (existsSync(venvPython)) {
    candidates.push({ command: venvPython, args: uvicornModuleArgs, label: ".venv python" });
  }
  candidates.push({ command: "python3", args: uvicornModuleArgs, label: "python3" });
  candidates.push({ command: "python", args: uvicornModuleArgs, label: "python" });
}

let child = null;
let selectedLabel = "";

function pipeSignal(signal) {
  if (child && !child.killed) {
    child.kill(signal);
  }
}

process.on("SIGINT", () => pipeSignal("SIGINT"));
process.on("SIGTERM", () => pipeSignal("SIGTERM"));

function launchCandidate(index) {
  if (index >= candidates.length) {
    console.error("[dev:rag] Could not start uvicorn. Install dependencies with: pip install -r rag/requirements.txt");
    process.exit(1);
  }

  const candidate = candidates[index];
  if (!candidate) {
    process.exit(1);
  }

  selectedLabel = candidate.label;
  console.log(
    `[dev:rag] Starting RAG API using ${candidate.label} on http://${host}:${port}${enableReload ? " (reload on)" : ""}`
  );

  child = spawn(candidate.command, candidate.args, {
    cwd,
    stdio: "inherit",
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1"
    }
  });

  child.on("error", (error) => {
    const code = error && typeof error === "object" && "code" in error ? error.code : "";
    if (code === "ENOENT") {
      console.warn(`[dev:rag] ${candidate.label} not found. Trying next interpreter.`);
      launchCandidate(index + 1);
      return;
    }

    const message = error instanceof Error ? error.message : String(error);
    console.error(`[dev:rag] Failed to start via ${candidate.label}: ${message}`);
    process.exit(1);
  });

  child.on("exit", (code, signal) => {
    if (code === 0) {
      process.exit(0);
      return;
    }

    if (signal) {
      process.exit(1);
      return;
    }

    if (index + 1 < candidates.length) {
      console.warn(`[dev:rag] ${selectedLabel} exited with code ${code}. Trying next interpreter.`);
      launchCandidate(index + 1);
      return;
    }

    console.error(`[dev:rag] RAG API exited with code ${code}.`);
    process.exit(code ?? 1);
  });
}

launchCandidate(0);
