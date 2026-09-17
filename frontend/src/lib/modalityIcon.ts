import type { ElementType } from "react";
import { AudioLines, FileText, Image, Link, Search, Video } from "lucide-react";
import type { Modality } from "../types";

export const modalityIcon: Record<Modality, ElementType> = {
  text: FileText,
  url: Link,
  pdf: FileText,
  image: Image,
  audio: AudioLines,
  video: Video,
  query: Search,
};
