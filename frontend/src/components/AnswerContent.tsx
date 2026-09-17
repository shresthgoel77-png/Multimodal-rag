function cleanAnswerText(value: string) {
  return value
    .replace(/\[[a-f0-9]{8,12}-\d+\]/gi, "")
    .replace(/\*\*/g, "")
    .replace(/^\s*\*\s+/gm, "- ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function AnswerContent({ answer }: { answer: string }) {
  const cleaned = cleanAnswerText(answer);

  if (!cleaned) {
    return <p>The answer will appear here after the ADK coordinator retrieves evidence from your sources.</p>;
  }

  const blocks = cleaned.split(/\n\s*\n/).filter(Boolean);
  return (
    <div className="answer-content">
      {blocks.map((block, blockIndex) => {
        const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
        const isList = lines.length > 1 && lines.every((line) => line.startsWith("- "));
        const hasHeadingAndList =
          lines.length > 2 &&
          lines[0].endsWith(":") &&
          lines.slice(1).every((line) => line.startsWith("- "));
        if (isList) {
          return (
            <ul key={blockIndex}>
              {lines.map((line, lineIndex) => <li key={lineIndex}>{line.replace(/^- /, "")}</li>)}
            </ul>
          );
        }
        if (hasHeadingAndList) {
          return (
            <div className="answer-section" key={blockIndex}>
              <h3>{lines[0].replace(/:$/, "")}</h3>
              <ul>
                {lines.slice(1).map((line, lineIndex) => <li key={lineIndex}>{line.replace(/^- /, "")}</li>)}
              </ul>
            </div>
          );
        }
        return lines.map((line, lineIndex) => {
          if (line.endsWith(":") && line.length < 48) {
            return <h3 key={`${blockIndex}-${lineIndex}`}>{line.replace(/:$/, "")}</h3>;
          }
          if (line.startsWith("- ")) {
            return <ul key={`${blockIndex}-${lineIndex}`}><li>{line.replace(/^- /, "")}</li></ul>;
          }
          return <p key={`${blockIndex}-${lineIndex}`}>{line}</p>;
        });
      })}
    </div>
  );
}
