"use client";
import { CollectionsProvider } from "@/lib/collections-context";
import { useCollections } from "@/lib/collections-context";
import AppSidebar from "@/components/AppSidebar";

function AppLayoutInner({ children }: { children: React.ReactNode }) {
  const { sidebarOpen, closeSidebar } = useCollections();

  return (
    <div className="flex h-screen overflow-hidden theme-bg theme-text relative">
      {/* Background glows */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-500/5 blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-purple-500/5 blur-[120px] pointer-events-none" />

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div className="drawer-overlay lg:hidden fixed inset-0 z-40 bg-black/60"
             onClick={closeSidebar} />
      )}

      {/* Sidebar — desktop: static | mobile: fixed drawer */}
      <div className={`
        shrink-0 transition-all duration-300 ease-in-out
        lg:relative lg:z-auto lg:translate-x-0
        fixed inset-y-0 left-0 z-50
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
      `}>
        <AppSidebar onClose={closeSidebar} />
      </div>

      {/* Main content */}
      <div className="flex flex-col flex-1 min-w-0 relative z-10">
        {children}
      </div>
    </div>
  );
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <CollectionsProvider>
      <AppLayoutInner>{children}</AppLayoutInner>
    </CollectionsProvider>
  );
}
