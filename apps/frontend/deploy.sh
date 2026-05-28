#!/usr/bin/env bash
set -euo pipefail

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required but not found in PATH."
  exit 1
fi

if ! command -v vercel >/dev/null 2>&1; then
  echo "Vercel CLI not found globally. Using npx vercel."
fi

if [ ! -f ".vercel/project.json" ]; then
  echo "Vercel project is not linked yet. Linking now."
  npx vercel link
fi

echo "Preparing SEO artifacts (robots.txt + sitemap.xml)..."
npm run seo:prepare

echo "Deploying frontend to Vercel production..."
npx vercel --prod --yes
