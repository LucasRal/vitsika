/** Same 8-slot categorical palette as scripts/viz.py (fixed order, never
 * cycled); the 9th+ category folds into "other". Slot order = category size. */
export const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
export const OTHER = "#b5b3ac";

/** POC subfamilies in size order (embeddings_index.csv), so colours are stable across pages. */
export const SUBFAMILY_ORDER = [
  "myrmicinae", "formicinae", "ponerinae", "dolichoderinae", "amblyoponinae",
  "pseudomyrmecinae", "dorylinae", "proceratiinae",
];

export function colourFor(category: string, order: string[]): string {
  const i = order.indexOf(category);
  return i >= 0 && i < PALETTE.length ? PALETTE[i] : OTHER;
}
export const subfamilyColour = (s: string) => colourFor(s, SUBFAMILY_ORDER);
