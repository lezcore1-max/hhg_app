import { useRef, useState, useCallback } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { askVoice, type AskResponse } from "@/lib/api";

type Stage = "idle" | "recording" | "processing";

interface MicOrbProps {
  onResult?: (result: AskResponse) => void;
  onError?: (msg: string) => void;
}

export function MicOrb({ onResult, onError }: MicOrbProps) {
  const [stage, setStage] = useState<Stage>("idle");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        setStage("processing");
        try {
          const result = await askVoice(blob);
          onResult?.(result);
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : "Unknown error";
          onError?.(msg);
        } finally {
          setStage("idle");
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setStage("recording");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Mic access denied";
      onError?.(msg);
    }
  }, [onResult, onError]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
  }, []);

  const handleClick = () => {
    if (stage === "idle") startRecording();
    else if (stage === "recording") stopRecording();
  };

  const ariaLabel =
    stage === "idle"
      ? "Tap to speak"
      : stage === "recording"
      ? "Stop recording"
      : "Processing…";

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={stage === "processing"}
      aria-label={ariaLabel}
      aria-pressed={stage === "recording"}
      className="group relative grid h-40 w-40 place-items-center rounded-full focus:outline-none"
    >
      {stage === "recording" && (
        <>
          <span className="absolute inset-0 animate-ping rounded-full border-2 border-primary/60" />
          <span className="absolute -inset-4 animate-pulse rounded-full border border-primary/40" />
        </>
      )}
      <span
        className={`relative grid h-28 w-28 place-items-center rounded-full bg-primary transition-transform duration-300 group-hover:scale-105 ${
          stage === "recording" ? "shadow-[0_0_70px_20px_rgba(242,201,76,0.35)]" : ""
        } ${stage === "processing" ? "opacity-60" : ""}`}
      >
        {stage === "processing" ? (
          <Loader2 className="h-10 w-10 animate-spin text-primary-foreground" aria-hidden />
        ) : stage === "recording" ? (
          <Square className="h-10 w-10 text-primary-foreground" aria-hidden />
        ) : (
          <Mic className="h-10 w-10 text-primary-foreground" aria-hidden />
        )}
      </span>
    </button>
  );
}
