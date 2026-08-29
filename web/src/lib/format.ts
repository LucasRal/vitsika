export const pct = (p: number, digits = 0) => `${(p * 100).toFixed(digits)}%`;
export const num = (n: number) => n.toLocaleString("en-US");
export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
/** "Camponotus imitator" -> "imitator" (species epithet without the genus). */
export const epithet = (species: string | null, genus: string) =>
  species ? species.replace(new RegExp(`^${genus}\\s+`), "") : null;
