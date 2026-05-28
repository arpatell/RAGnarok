const loadedImageUrls = new Set<string>();
const loadingImagePromises = new Map<string, Promise<boolean>>();

export function markImageLoaded(url: string) {
  const trimmed = url.trim();
  if (!trimmed) {
    return;
  }

  loadedImageUrls.add(trimmed);
  loadingImagePromises.delete(trimmed);
}

export function isImageLoaded(url: string): boolean {
  const trimmed = url.trim();
  return Boolean(trimmed && loadedImageUrls.has(trimmed));
}

export function isImageLoading(url: string): boolean {
  const trimmed = url.trim();
  return Boolean(trimmed && loadingImagePromises.has(trimmed));
}

export async function preloadImageOnce(
  url: string,
  onLoaded?: (startedAt: number) => void
): Promise<boolean> {
  const trimmed = url.trim();
  if (!trimmed) {
    return false;
  }

  if (loadedImageUrls.has(trimmed)) {
    return true;
  }

  const existing = loadingImagePromises.get(trimmed);
  if (existing) {
    return existing;
  }

  const startedAt = performance.now();
  const promise = new Promise<boolean>((resolve) => {
    const image = new Image();
    image.onload = () => {
      markImageLoaded(trimmed);
      onLoaded?.(startedAt);
      resolve(true);
    };
    image.onerror = () => {
      loadingImagePromises.delete(trimmed);
      resolve(false);
    };
    image.src = trimmed;
  });

  loadingImagePromises.set(trimmed, promise);
  return promise;
}
