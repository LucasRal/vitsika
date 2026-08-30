import { describe, expect, it } from "vitest";
import { LOW_CONFIDENCE, SUPPORT_MIN, SUPPORT_SIMILARITY, verdict } from "./verdict";
import type { AnalyzeResponse } from "./api";
// Payloads captured from the production API (POST /api/analyze) on 2026-08-30,
// after temperature calibration of the probe (scripts/07_eval.py, api/probe.py).
// Both real payloads are now "confident"; the other tiers are derived from
// them by scaling the top-1 probability, which is exactly what calibration
// does not change (the ranking and the neighbours).
import royidris from "./__fixtures__/analyze_royidris_casent0002219.json";
import tetraponera from "./__fixtures__/analyze_tetraponera_casent0012838.json";

const R = royidris as AnalyzeResponse;
const T = tetraponera as AnalyzeResponse;

describe("verdict tiers", () => {
  it("Royidris quick example: calibrated top-1 99.7 % → confident (no banner)", () => {
    const v = verdict(R);
    expect(v.genus).toBe("Royidris");
    expect(v.top1).toBeGreaterThan(0.99);
    expect(v.support).toBe(3); // 3 Royidris ≥ 0.85; Monomorium 0.861 and Nesomyrmex 0.858 don't count
    expect(v.tier).toBe("confident");
  });

  it("Tetraponera held-out casent0012838: calibrated top-1 99.9 %, all 5 neighbours agree → confident", () => {
    const v = verdict(T);
    expect(v.genus).toBe("Tetraponera");
    expect(v.support).toBe(5);
    expect(v.tier).toBe("confident");
  });

  it("modest top-1 with ≥ 3 supporting neighbours → supported (neutral note)", () => {
    const modest = (x: AnalyzeResponse, p: number) => ({ ...x, predictions: [{ ...x.predictions[0], probability: p }, ...x.predictions.slice(1)] });
    expect(verdict(modest(R, 0.37))).toMatchObject({ tier: "supported", support: 3 });
    expect(verdict(modest(T, 0.49))).toMatchObject({ tier: "supported", support: 5 });
  });

  it("top-1 ≥ 0.5 → confident, whatever the neighbours say", () => {
    const bumped = { ...T, predictions: [{ ...T.predictions[0], probability: 0.5 }, ...T.predictions.slice(1)] };
    expect(verdict(bumped).tier).toBe("confident");
    const noSupport = { ...bumped, similar: T.similar.map((s) => ({ ...s, genus: "Camponotus" })) };
    expect(verdict(noSupport).tier).toBe("confident");
  });

  it("modest top-1 and fewer than 3 supporting neighbours → low (amber banner)", () => {
    // Royidris payload at a modest top-1, with its 3rd Royidris neighbour relabelled: support drops to 2.
    const Rm = { ...R, predictions: [{ ...R.predictions[0], probability: 0.37 }, ...R.predictions.slice(1)] };
    const similar = Rm.similar.map((s, i) => (i === 2 ? { ...s, genus: "Monomorium" } : s));
    const v = verdict({ ...Rm, similar });
    expect(v.support).toBe(SUPPORT_MIN - 1);
    expect(v.tier).toBe("low");
    // Same genus but weak similarity does not count either.
    const weak = Rm.similar.map((s) => ({ ...s, similarity: SUPPORT_SIMILARITY - 0.01 }));
    expect(verdict({ ...Rm, similar: weak })).toMatchObject({ support: 0, tier: "low" });
  });

  it("supports exactly at the thresholds", () => {
    const Rm = { ...R, predictions: [{ ...R.predictions[0], probability: LOW_CONFIDENCE - 0.001 }, ...R.predictions.slice(1)] };
    const edge = Rm.similar.map((s, i) => (i < 3 ? { ...s, genus: "Royidris", similarity: SUPPORT_SIMILARITY } : s));
    expect(verdict({ ...Rm, similar: edge }).tier).toBe("supported");
  });

  it("handles an empty answer without throwing", () => {
    expect(verdict({ predictions: [], similar: [] })).toMatchObject({ tier: "low", support: 0, top1: 0 });
  });
});
