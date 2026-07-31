// Mirrors _DEVIATION_PATTERNS in prorag/chat/citations.py: (S1), [s1],
// [source 1] and [1] are all forms the answer model actually emits.
//
// This has to exist client-side because the server normalizes those forms only
// AFTER the stream ends (chat/router.py persists the normalized text but
// streams the RAW tokens). Matching only /\[S\d+\]/ meant a model writing
// "(S1)" showed dead plain text while answering, then silently became a
// clickable pill on reload — the same answer rendered two different ways.
//
// Kept in a .ts module rather than beside the component so it stays testable
// (node --experimental-strip-types can't load .tsx).
const MARKER_SOURCE = String.raw`(\(S\d+\)|\[(?:source\s+|s)?\d+\])`;

/** Fresh object per call: a shared /g regex carries `lastIndex` between uses. */
export const markerSplitter = () => new RegExp(MARKER_SOURCE, "gi");

export const isCitationMarker = (part: string) => new RegExp(`^${MARKER_SOURCE}$`, "i").test(part);

/** Splits answer text into prose and citation markers, dropping empty pieces. */
export const splitOnCitations = (text: string) => text.split(markerSplitter()).filter(Boolean);

/** The 1-based source index inside a marker of any supported form. */
export const markerIndex = (marker: string) => Number(marker.match(/\d+/)![0]);
