"use client";
import {
  createContext, useCallback, useContext, useEffect, useState, type ReactNode,
} from "react";
import type { Collection } from "./types";
import { fetchCollections } from "./api";

const CACHE_KEY = "yourag_collections_v1";
const THEME_KEY = "yourag_theme";

function readCache(): Collection[] {
  try { return JSON.parse(localStorage.getItem(CACHE_KEY) ?? "[]"); } catch { return []; }
}

interface CollectionsCtx {
  collections: Collection[];
  selectedCollections: Collection[];
  activeVideo: Collection | null;
  loadingCollections: boolean;
  apiOnline: boolean;
  theme: "dark" | "light";
  setTheme: (t: "dark" | "light") => void;
  setSelectedCollections: React.Dispatch<React.SetStateAction<Collection[]>>;
  setActiveVideo: React.Dispatch<React.SetStateAction<Collection | null>>;
  loadCollections: () => Promise<void>;
  onDeleted: (name: string) => void;
}

const Ctx = createContext<CollectionsCtx | null>(null);

export function CollectionsProvider({ children }: { children: ReactNode }) {
  const [collections, setCollections] = useState<Collection[]>(() =>
    typeof window !== "undefined" ? readCache() : []
  );
  const [selectedCollections, setSelectedCollections] = useState<Collection[]>([]);
  const [activeVideo, setActiveVideo] = useState<Collection | null>(null);
  const [loadingCollections, setLoadingCollections] = useState(true);
  const [apiOnline, setApiOnline] = useState<boolean>(() =>
    typeof window !== "undefined" ? readCache().length > 0 : false
  );
  const [theme, setThemeState] = useState<"dark" | "light">(() => {
    if (typeof window !== "undefined") {
      return (localStorage.getItem(THEME_KEY) as "dark" | "light") ?? "dark";
    }
    return "dark";
  });

  const setTheme = useCallback((t: "dark" | "light") => {
    setThemeState(t);
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem(THEME_KEY, t); } catch {}
  }, []);

  // Apply theme on mount
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const loadCollections = useCallback(async () => {
    setLoadingCollections(true);
    for (let attempt = 0; attempt < 5; attempt++) {
      try {
        const data = await fetchCollections();
        setCollections(data);
        setApiOnline(true);
        setLoadingCollections(false);
        try { localStorage.setItem(CACHE_KEY, JSON.stringify(data)); } catch {}
        return;
      } catch {
        if (attempt < 4) await new Promise(r => setTimeout(r, 1_000 * (attempt + 1)));
      }
    }
    setApiOnline(false);
    setLoadingCollections(false);
  }, []);

  const onDeleted = useCallback((name: string) => {
    setCollections(prev => prev.filter(c => c.name !== name));
    setSelectedCollections(prev => prev.filter(c => c.name !== name));
    setActiveVideo(prev => (prev?.name === name ? null : prev));
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify(readCache().filter(c => c.name !== name)));
    } catch {}
  }, []);

  useEffect(() => { loadCollections(); }, [loadCollections]);

  return (
    <Ctx.Provider value={{
      collections, selectedCollections, activeVideo, loadingCollections, apiOnline, theme,
      setTheme, setSelectedCollections, setActiveVideo, loadCollections, onDeleted,
    }}>
      {children}
    </Ctx.Provider>
  );
}

export function useCollections() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useCollections must be used within CollectionsProvider");
  return ctx;
}
