# Severity rules (Low / Medium / High)

Severity is decided by fixed, documented rules from the defect's **type**, **size** and **location**. The detector's **confidence is not used**: confidence says how sure the model is that a defect is there, not how bad it is.

| Input | How it is measured |
|---|---|
| Type | The detected class. `crazing`, `pitted_surface` and `inclusion` are treated as **critical** because cracks, pits and embedded particles can grow or weaken the strip. `patches`, `rolled-in_scale` and `scratches` are treated as cosmetic or reworkable unless large. |
| Size | Box area as a percentage of the image area (`area_pct`). |
| Location | `edge` if the box comes within 10% of any image border, else `centre`. Edge defects matter more because the strip edge is where cracks start during later forming and slitting. |

## Rules (applied per defect, in this order)

1. **High**: critical type **and** area > 5%.
2. **Medium**: area > 2% (any type), **or** critical type at the edge.
3. **Low**: everything else.

The image (part) gets the worst severity of its defects. Verdict is rule-based: any detection means **FAIL**, none means **PASS**.

Operator action from the rules (used when Llama is off or fails):
- High: stop and quarantine the part, check the upstream process.
- Medium: hold for manual inspection, check the line.
- Low: pass with a note, watch the next parts.

## Code
`SEVERITY_RULES` and `severity_for()` in `pipeline/qc_inspector.py`.

## Limits
Thresholds are a documented starting point in % of image area. NEU-DET does not publish a physical scale (mm per pixel), so they are not tied to a millimetre tolerance. On a real line they should be set from the plant's acceptance standard and the camera's calibrated field of view.
