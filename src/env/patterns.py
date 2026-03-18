import torch


def _normalize(offsets):
    min_r = min(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    return [(r - min_r, c - min_c) for r, c in offsets]


def _project_to_direction(offsets, dr, dc):
    projected = []
    for _, t in offsets:
        projected.append((dr * t, dc * t))
    return _normalize(projected)


def build_masks(window: int = 5):
    patterns = {
        "five": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],  # XXXXX
        "open_four": [(0, 1), (0, 2), (0, 3), (0, 4)],  # _XXXX
        "jump_four": [(0, 0), (0, 1), (0, 2), (0, 4)],  # XXX_X
        "open_three": [(0, 1), (0, 2), (0, 3)],  # _XXX_
        "half_open_three": [(0, 0), (0, 1), (0, 2)],  # XXX__
        "jump_three": [(0, 0), (0, 1), (0, 3)],  # XX_X_
    }
    directions = {
        "h": (0, 1),  # horizontal
        "v": (1, 0),  # vertical
        "d1": (1, 1),  # diagonal
        "d2": (1, -1),  # anti-diagonal
    }

    masks = []
    names = []
    seen = set()
    for p_name, offsets in patterns.items():
        for d_name, (dr, dc) in directions.items():
            transformed = _project_to_direction(offsets, dr, dc)
            key = frozenset(transformed)

            if key not in seen:
                seen.add(key)
                grid = torch.zeros(window, window)
                for r, c in transformed:
                    if 0 <= r < window and 0 <= c < window:
                        grid[r, c] = 1.0
                masks.append(grid.flatten())
                names.append(f"{p_name}_{d_name}")

        for d_name, (dr, dc) in directions.items():
            transformed = _project_to_direction(offsets, -dr, -dc)
            key = frozenset(transformed)

            if key not in seen:
                seen.add(key)
                grid = torch.zeros(window, window)
                for r, c in transformed:
                    if 0 <= r < window and 0 <= c < window:
                        grid[r, c] = 1.0
                masks.append(grid.flatten())
                names.append(f"{p_name}_{d_name}_rev")

    return torch.stack(masks), names


if __name__ == "__main__":
    window = 5
    masks, names = build_masks(window)
    masks = masks.view(-1, window, window)
    for i, (mask, name) in enumerate(zip(masks, names)):
        print(f"[{i}] mask {name}")
        for row in mask:
            line = [" ".join("X" if v.item() == 1 else "_" for v in row)]
            print(line)
