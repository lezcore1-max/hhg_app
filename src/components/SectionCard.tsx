import type { ReactNode } from "react";

export function SectionCard({
  index,
  title,
  children,
}: {
  index: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="grain rounded-lg border border-border bg-card p-6 sm:p-8">
      <p className="label-mono text-primary">{index}</p>
      <h3 className="mt-3 text-3xl text-card-foreground">{title}</h3>
      <div className="mt-3 text-sm leading-relaxed text-muted-foreground">{children}</div>
    </div>
  );
}
