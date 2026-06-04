/**
 * Seeded pseudo-random number generator for reproducible boards (파일트리: rng.ts).
 *
 * mulberry32 — a small, fast 32-bit PRNG with a single 32-bit state word. Chosen for
 * reproducibility (same seed → same sequence) and because it is trivial to re-implement
 * identically in Python (server/trainer) if mine layouts ever need to match cross-language.
 */

/** A function returning the next float in [0, 1). */
export type Rng = () => number;

/**
 * Create a seeded RNG. Deterministic: a given `seed` always yields the same sequence.
 * @param seed any integer; coerced to a uint32.
 */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Random integer in [0, max) using the given RNG.
 * @param max exclusive upper bound (must be > 0).
 */
export function randInt(rng: Rng, max: number): number {
  return Math.floor(rng() * max);
}
