import re
import argparse
from pathlib import Path

def adjust_bounds(lines, k):
    pattern = re.compile(r'\(assert \((<=|>=) (X_\d+) ([\d\.\-eE]+)\)\)')
    bounds = {}
    line_map = {}

    # First pass: identify bounds
    for idx, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            op, var, val = match.groups()
            val = float(val)
            if var not in bounds:
                bounds[var] = {}
                line_map[var] = {}
            bounds[var][op] = val
            line_map[var][op] = idx

    # Second pass: update bounds
    updated_lines = lines.copy()
    for var in bounds:
        if "<=" in bounds[var] and ">=" in bounds[var]:
            lb0 = bounds[var]['>=']
            ub0 = bounds[var]['<=']
            mid = 0.5 * (lb0 + ub0)
            delta = 0.5 * (ub0 - lb0)
            new_lb = mid - k * delta
            new_ub = mid + k * delta

            updated_lines[line_map[var]['<=']] = f"(assert (<= {var} {new_ub}))"
            updated_lines[line_map[var]['>=']] = f"(assert (>= {var} {new_lb}))"

    return updated_lines

def main():
    parser = argparse.ArgumentParser(description="Adjust bounds in a .vnnlib file using a scaling factor k.")
    parser.add_argument("input", help="Path to the input .vnnlib file")
    parser.add_argument("k", type=float, help="Scaling factor k (e.g., 1.0, 2.5)")
    parser.add_argument("--output", help="Path to save the updated .vnnlib file (optional)")

    args = parser.parse_args()
    input_path = Path(args.input)
    k = args.k

    lines = input_path.read_text().splitlines()
    updated_lines = adjust_bounds(lines, k)

    output_path = Path(args.output) if args.output else input_path.with_stem(f"{input_path.stem}_k{k}")
    output_path.write_text("\n".join(updated_lines))
    print(f"Updated file saved to: {output_path}")

if __name__ == "__main__":
    main()
