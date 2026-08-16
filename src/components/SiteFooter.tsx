import { Link } from "@tanstack/react-router";
import { nav } from "@/lib/site-data";

export function SiteFooter() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-5 py-10 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-display text-2xl text-foreground">
            Tilt<span className="text-primary">.</span>
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Voice-native RAG · Hacker House Goa 2026
          </p>
        </div>
        <nav className="flex flex-wrap gap-4">
          {nav.map((n) => (
            <Link
              key={n.to}
              to={n.to}
              className="label-mono text-muted-foreground transition-colors hover:text-foreground"
            >
              {n.label}
            </Link>
          ))}
        </nav>
        <p className="text-xs text-muted-foreground">© 2026 Tilt · HH Goa 2026</p>
      </div>
    </footer>
  );
}
