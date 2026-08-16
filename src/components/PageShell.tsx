import type { ReactNode } from "react";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";

export function PageShell({
  eyebrow,
  title,
  intro,
  children,
}: {
  eyebrow: string;
  title: string;
  intro: string;
  children: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <SiteHeader />
      <main className="mx-auto max-w-6xl px-5 py-16 sm:py-24">
        <p className="label-mono text-primary">{eyebrow}</p>
        <h1 className="mt-4 text-6xl tracking-tight sm:text-8xl">{title}</h1>
        <p className="mt-6 max-w-2xl text-sm leading-relaxed text-muted-foreground sm:text-base">
          {intro}
        </p>
        <div className="mt-12">{children}</div>
      </main>
      <SiteFooter />
    </div>
  );
}
