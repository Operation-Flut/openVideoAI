/**
 * Feature display names.
 *
 * The single source of truth is `models.json`, which declares a `displayName`
 * for every feature and `arch`/`defaultFor` for every model. The API exposes
 * that metadata and `useFeatures()` publishes it here, so the UI never keeps a
 * second, hardcoded copy of feature labels or model aliases.
 *
 * The names are stored in a tiny external store rather than module state so
 * components re-render when the catalog arrives (labels are resolved during
 * render, including inside `useMemo` blocks).
 */
import { useSyncExternalStore } from 'react'

let featureDisplayNames: Record<string, string> = {}
const listeners = new Set<() => void>()

/** Store the catalog's feature display names (called when the catalog loads). */
export function setFeatureDisplayNames(names: Record<string, string>): void {
  featureDisplayNames = names
  listeners.forEach((listener) => listener())
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

function getSnapshot(): Record<string, string> {
  return featureDisplayNames
}

/**
 * Reactive catalog labels. Subscribe with this (and include the result in any
 * `useMemo` dependency list that calls `displayFeature`) to re-render when the
 * catalog arrives.
 */
export function useFeatureDisplayNames(): Record<string, string> {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
}

/** User-facing label for a feature key, falling back to the raw key. */
export function displayFeature(name: string): string {
  return featureDisplayNames[name] || name
}
