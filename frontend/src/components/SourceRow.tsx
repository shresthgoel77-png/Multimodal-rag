import { Box, Loader2, Trash2 } from "lucide-react";
import type { RackSource } from "../types";
import { modalityIcon } from "../lib/modalityIcon";

export function SourceRow({
  source,
  isRemoving,
  onRemove,
}: {
  source: RackSource;
  isRemoving: boolean;
  onRemove: (source: RackSource) => void;
}) {
  const Icon = modalityIcon[source.modality] || Box;
  const embeddingPath = String(source.metadata?.embedding_path || "");
  return (
    <div className="source-row">
      <div className={`modality-dot ${source.modality}`}>
        <Icon size={15} />
      </div>
      <div className="source-copy">
        <div className="source-title">{source.title}</div>
        <div className="source-summary">{source.summary}</div>
        {embeddingPath && <div className="source-meta">{embeddingPath.replace("gemini-", "Gemini ")}</div>}
      </div>
      <div className="source-actions">
        <button className="delete-source" onClick={() => onRemove(source)} disabled={isRemoving} aria-label={`Remove ${source.title}`}>
          {isRemoving ? <Loader2 className="spin" size={13} /> : <Trash2 size={13} />}
        </button>
      </div>
    </div>
  );
}
