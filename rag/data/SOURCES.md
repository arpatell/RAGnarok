# Data Sources

The current MAL API v2 catalog remains the authority for title IDs, title identity,
current metadata, and covers. Bulk snapshots fill missing fields and cast links;
they do not replace newer official API values. Shared character IDs do not imply
that an anime cast can be copied onto its manga adaptation.

Offline bulk inputs used for enrichment:

| Publisher / Dataset | Pinned Version | Publisher-Declared License |
| --- | --- | --- |
| [syahrulapriansyah2 / Anime Characters](https://www.kaggle.com/datasets/syahrulapriansyah2/anime-characters) | 1 | MIT |
| [syahrulapriansyah2 / MyAnimeList 2025](https://www.kaggle.com/datasets/syahrulapriansyah2/myanimelist-2025) | 1 | MIT |
| [hamzaashfaque1999 / MyAnimeList Scraped Data](https://www.kaggle.com/datasets/hamzaashfaque1999/myanimelist-scraped-data) | 1 | MIT |
| [andreuvallhernndez / MyAnimeList Jikan Database](https://www.kaggle.com/datasets/andreuvallhernndez/myanimelist-jikan) | 1 | CC0: Public Domain |

Underlying entries originate from MyAnimeList and its contributors. These are
historical snapshots, not proof of current, exhaustive MAL field coverage.
Profiles directly fetched from MAL are preserved over bulk versions. Empty
biography placeholders are not counted as biographies.

The importer records source URLs, versions, publisher license declarations,
publication timestamps, byte counts, and SHA-256 hashes in ignored local manifests.
Raw archives and profile exports are not deployed. New indexed material uses
the same embedding model, sparse vocabulary, index, and namespace as the existing
catalog. No additional serving service is required.

## MIT Notice

Copyright belongs to the dataset publishers and original contributors listed above.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
