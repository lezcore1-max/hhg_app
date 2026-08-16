/**
 * Reports a runtime error to the configured error-tracking backend.
 *
 * This is a safe no-op by default so the app runs without any external
 * service. To enable real tracking, wire up your provider (Sentry,
 * Highlight, etc.) inside this function.
 */
export function reportLovableError(
  error: unknown,
  info: Record<string, unknown> = {},
): void {
  if (import.meta.env.DEV) {
    console.error("[error-boundary]", info, error)
  }
}
