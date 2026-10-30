import networkx as nx
import numpy as np
from scipy.interpolate import splev, splprep


def extract_disjoint_paths(G: nx.Graph, min_path_length=5, avg_window=10):
    """
    Traverse the skeleton graph and extract all disjoint paths (non-overlapping edge chains),
    using a moving average of previous direction vectors for smoother traversal.
    """
    visited_edges = set()
    paths = []

    for node in G.nodes():
        for neighbor in G.neighbors(node):
            edge = tuple(sorted((node, neighbor)))
            if edge in visited_edges:
                continue

            path = [node, neighbor]
            visited_edges.add(edge)

            current = neighbor
            prev = node
            history = [np.array(current) - np.array(prev)]

            while True:
                neighbors = [n for n in G.neighbors(current) if n != prev]
                candidates = []

                # Compute average direction from history
                avg_dir = np.mean(history[-avg_window:], axis=0)
                norm = np.linalg.norm(avg_dir)
                if norm < 1e-6:
                    break
                avg_dir /= norm

                for n in neighbors:
                    e = tuple(sorted((current, n)))
                    if e in visited_edges:
                        continue
                    next_vec = np.array(n) - np.array(current)
                    norm = np.linalg.norm(next_vec)
                    if norm < 1e-6:
                        continue
                    next_vec = next_vec.astype(np.float64)
                    next_vec /= norm
                    angle = np.arccos(np.clip(np.dot(avg_dir, next_vec), -1.0, 1.0))
                    candidates.append((angle, n, next_vec, e))

                if candidates:
                    candidates.sort()
                    _, best_n, best_vec, best_edge = candidates[0]
                    visited_edges.add(best_edge)
                    path.append(best_n)
                    history.append(best_vec)
                    prev, current = current, best_n
                else:
                    break

            if len(path) >= min_path_length:
                paths.append(path)

    merged_paths = merge_paths(paths, require_direction=False)
    clean_paths = remove_redundant_paths(merged_paths)
    return clean_paths


def compute_angle(p1, p2):
    v = np.array(p2) - np.array(p1)
    norm = np.linalg.norm(v)
    if norm == 0:
        return None
    return v / norm


def merge_paths(paths, dist_thresh=9.0, angle_thresh=np.pi / 6, require_direction=True):
    """
    Merge paths whose endpoints are close.
    If require_direction=True, also require direction to align (like try_merge_paths).
    If False, merge as long as distance is within threshold (like connect_close_paths).
    """
    merged = []
    used = [False] * len(paths)

    for i, p1 in enumerate(paths):
        if used[i]:
            continue

        path = p1[:]
        used[i] = True
        changed = True

        while changed:
            changed = False
            for j, p2 in enumerate(paths):
                if used[j] or i == j:
                    continue

                options = [(path, p2), (path, p2[::-1]), (p2[::-1], path), (p2, path)]
                for left, right in options:
                    if len(left) < 2 or len(right) < 2:
                        continue
                    dist = np.linalg.norm(np.array(left[-1]) - np.array(right[0]))
                    if dist > dist_thresh:
                        continue

                    if require_direction:
                        v1 = compute_angle(left[-2], left[-1])
                        v2 = compute_angle(right[0], right[1])
                        if v1 is None or v2 is None:
                            continue
                        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                        if angle > angle_thresh:
                            continue

                    path = left + (right[1:] if dist == 0 else right)

                    used[j] = True
                    changed = True
                    break
                if changed:
                    break

        merged.append(path)

    return merged


def remove_redundant_paths(paths, overlap_thresh=0.4):
    """
    Remove paths that are mostly overlapping with others (based on pixel occupancy).
    """
    pixel_map = {}
    path_pixel_sets = []

    for i, path in enumerate(paths):
        pixels = set()
        for x, y in path:
            px = int(round(x))
            py = int(round(y))
            pixels.add((px, py))
            pixel_map.setdefault((px, py), set()).add(i)
        path_pixel_sets.append(pixels)

    keep = [True] * len(paths)
    for i, pixels in enumerate(path_pixel_sets):
        if not keep[i]:
            continue
        overlapping = sum(1 for p in pixels if any(other != i and keep[other] for other in pixel_map[p]))
        if overlapping / max(1, len(pixels)) >= overlap_thresh:
            keep[i] = False

    return [p for k, p in zip(keep, paths) if k]


import numpy as np


def sort_paths_human_like_final(
    paths,
    size_threshold_ratio=0.15,
    size_metric_ratio=0.7,
    num_vertical_bins=10,
    alpha=1.0,
    beta=3.0,
    gamma=5.0,
    delta=0.5,
):
    """
    通过分阶段、分区域的策略，模拟人类绘画顺序。
    - 阶段1: 绘制大型笔画，其“大小”由直径和面积共同决定。
    - 阶段2: 将细节笔画按垂直位置分箱，从上到下逐个区域绘制，
             每个区域内部遵循就近原则。

    Args:
        paths: list of paths (each a list of (x,y))
        size_threshold_ratio: 决定笔画大小分类的阈值比例。
        size_metric_ratio: 直径在“大小”指标中的权重 (0.0 to 1.0)。
        num_vertical_bins: 在阶段2中，将细节垂直切分为多少个区域。
        alpha: 路径长度的权重。
        beta: 与上一笔画距离的惩罚权重。
        gamma: “大小”指标的权重。
        delta: 离全局中心距离的权重 (奖励外围笔画)。
    """
    if not paths:
        return []

    # --- 步骤 1: 预计算所有路径的属性 ---
    path_props = []
    all_points = np.concatenate([np.array(p) for p in paths if np.array(p).size > 0])
    if all_points.size == 0:
        return []

    global_ymin, global_ymax = all_points[:, 1].min(), all_points[:, 1].max()
    global_height = global_ymax - global_ymin if global_ymax > global_ymin else 1.0
    global_center = np.mean(all_points, axis=0)

    def path_length(path):
        if len(path) < 2:
            return 0
        return sum(
            np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
            for i in range(len(path) - 1)
        )

    for path in paths:
        arr = np.array(path)
        if arr.shape[0] < 2:
            continue

        xmin, ymin = arr.min(axis=0)
        xmax, ymax = arr.max(axis=0)

        width, height = xmax - xmin, ymax - ymin
        area = width * height
        diameter = np.sqrt(width**2 + height**2)
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

        path_props.append(
            {
                "path": path,
                "len": path_length(path),
                "area": area,
                "diameter": diameter,
                "center_dist": np.linalg.norm(center - global_center),
                "center_y": center[1],
            }
        )

    if not path_props:
        return []

    # 归一化并计算综合大小指标
    max_len = max(p["len"] for p in path_props) or 1.0
    max_area = max(p["area"] for p in path_props) or 1.0
    max_diameter = max(p["diameter"] for p in path_props) or 1.0
    max_center_dist = max(p["center_dist"] for p in path_props) or 1.0

    for p in path_props:
        p["norm_len"] = p["len"] / max_len
        p["norm_area"] = p["area"] / max_area
        p["norm_diameter"] = p["diameter"] / max_diameter
        p["norm_center_dist"] = p["center_dist"] / max_center_dist
        p["norm_size"] = (
            size_metric_ratio * p["norm_diameter"]
            + (1 - size_metric_ratio) * p["norm_area"]
        )

    # --- 步骤 2: 根据综合大小阈值将路径分为两组 ---
    size_threshold = size_threshold_ratio
    large_props = [p for p in path_props if p["norm_size"] >= size_threshold]
    small_props = [p for p in path_props if p["norm_size"] < size_threshold]

    ordered_paths = []
    last_point = global_center  # 初始参考点设为全局中心

    # --- 阶段 1: 绘制所有“大型”轮廓笔画 (逻辑与之前版本相同) ---
    if large_props:
        # (这部分贪心逻辑与上一版本完全相同，只是使用 norm_size)
        # 1a: 找到最佳起点
        start_scores = [
            gamma * p["norm_size"]
            + alpha * p["norm_len"]
            + delta * p["norm_center_dist"]
            for p in large_props
        ]
        start_idx = np.argmax(start_scores)
        first_path_prop = large_props.pop(start_idx)
        ordered_paths.append(first_path_prop["path"])
        last_point = np.array(ordered_paths[-1][-1])
        # 1b: 贪心绘制剩余部分
        remaining_large = large_props
        while remaining_large:
            best_score = -float("inf")
            best_idx = -1
            best_reverse = False
            for i, prop in enumerate(remaining_large):
                intrinsic_score = (
                    gamma * prop["norm_size"]
                    + alpha * prop["norm_len"]
                    + delta * prop["norm_center_dist"]
                )
                for check_reverse, endpoint in [
                    (False, prop["path"][0]),
                    (True, prop["path"][-1]),
                ]:
                    dist = np.linalg.norm(np.array(endpoint) - last_point)
                    score = intrinsic_score - beta * dist
                    if score > best_score:
                        best_score = score
                        best_idx = i
                        best_reverse = check_reverse
            next_prop = remaining_large.pop(best_idx)
            next_path = next_prop["path"]
            if best_reverse:
                next_path = next_path[::-1]
            ordered_paths.append(next_path)
            last_point = np.array(ordered_paths[-1][-1])

    # --- 阶段 2: 按垂直区域绘制所有“小型”细节笔画 ---
    if small_props:
        # 2a: 将所有小型笔画分配到垂直的“箱子”里
        bins = [[] for _ in range(num_vertical_bins)]
        for p in small_props:
            # 根据中心点的 Y 坐标决定它属于哪个箱子
            bin_index = int(
                ((p["center_y"] - global_ymin) / global_height)
                * (num_vertical_bins - 1)
            )
            bins[bin_index].append(p)

        # 2b: 从上到下遍历每个箱子
        for bin_content in bins:
            if not bin_content:
                continue  # 跳过空箱子

            # 在当前箱子内，进行局部的、纯粹基于就近原则的贪心绘制
            while bin_content:
                best_dist = float("inf")
                best_idx = -1
                best_reverse = False

                # 找到箱子中离上一个点最近的笔画作为下一个目标
                for i, prop in enumerate(bin_content):
                    for check_reverse, endpoint in [
                        (False, prop["path"][0]),
                        (True, prop["path"][-1]),
                    ]:
                        dist = np.linalg.norm(np.array(endpoint) - last_point)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                            best_reverse = check_reverse

                # 绘制这个最近的笔画
                next_prop = bin_content.pop(best_idx)
                next_path = next_prop["path"]
                if best_reverse:
                    next_path = next_path[::-1]

                ordered_paths.append(next_path)
                last_point = np.array(ordered_paths[-1][-1])

    return ordered_paths


def fit_bezier_curve(points, tolerance=0.5, num_points_multiplier=2):
    """
    Fit a Bezier curve to the points using spline interpolation.

    Args:
        points: List of (x, y) points
        tolerance: Smoothing tolerance (lower = smoother)
        num_points_multiplier: How many points to generate relative to input

    Returns:
        Smoothed points along the Bezier curve
    """
    if len(points) < 4:
        return points  # Not enough points for proper curve fitting

    points_array = np.array(points)

    try:
        # Fit a B-spline to the points
        tck, u = splprep(points_array.T, s=tolerance, k=3)

        # Generate more points along the curve for smoother result
        num_new_points = len(points) * num_points_multiplier
        new_u = np.linspace(0, 1, num_new_points)

        # Evaluate the spline at new points
        new_points = splev(new_u, tck)
        smoothed_points = list(zip(new_points[0], new_points[1]))

        return smoothed_points

    except Exception as e:
        print(f"Bezier curve fitting failed: {e}")
        return points  # Return original points if fitting fails


def is_circular_path(path, circular_threshold=0.2):
    """
    Detect if a path is approximately circular.

    Args:
        path: List of (x, y) points
        circular_threshold: Maximum allowed std/mean ratio for distances

    Returns:
        Boolean indicating if path is circular
    """
    if len(path) < 8:  # Too short to be a meaningful circle
        return False

    points = np.array(path)
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # Circular paths should have relatively uniform distances from center
    if mean_dist < 1e-6:  # Avoid division by zero
        return False

    return (std_dist / mean_dist) < circular_threshold


def extract_paths_from_skeleton(
    skeleton: np.ndarray,
    spacing: float = None,
    total_points: int = None,
    sort_paths: bool = True,
    min_path_length: int = 5,
    max_jump: float = None,
    smooth_circular_paths: bool = True,  # 新增参数：是否平滑圆形路径
    curve_smoothing_tolerance: float = 0.3,  # 新增参数：曲线平滑度
) -> list[list[tuple[float, float]]]:
    """
    Extract path sequences from a skeleton image with optional Bezier curve smoothing.

    Args:
        skeleton (np.ndarray): Binary skeleton image (bool or 0/1)
        spacing (float): Fixed sampling distance per path
        total_points (int): Total number of sampled points (if spacing is None)
        sort_paths (bool): Whether to sort the paths top-left to bottom-right
        min_path_length (int): Minimum pixel count for a valid path
        max_jump (float): Insert pen-up when path distance is large
        smooth_circular_paths (bool): Whether to apply Bezier smoothing to circular paths
        curve_smoothing_tolerance (float): Smoothing tolerance (lower = smoother)

    Returns:
        List of paths, each path is a list of (x, y) points
    """
    G = nx.Graph()
    h, w = skeleton.shape

    # Step 1: build graph from pixels
    for y in range(h):
        for x in range(w):
            if skeleton[y, x]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx_ = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                            G.add_edge((x, y), (nx_, ny))

    # Step 2: extract raw paths
    raw_paths = extract_disjoint_paths(G, min_path_length=min_path_length)

    # Step 3: sample points from paths
    def path_length(p):
        return sum(
            np.linalg.norm(np.array(p[i]) - np.array(p[i + 1]))
            for i in range(len(p) - 1)
        )

    total_len = sum(path_length(p) for p in raw_paths)
    sampled_paths = []

    for path in raw_paths:
        pts = [np.array(p, dtype=np.float32) for p in path]
        plen = path_length(pts)
        n_points = (
            2
            if plen < 1
            else (
                max(2, int(np.ceil(plen / spacing)) + 1)
                if spacing
                else max(2, int((plen / total_len) * total_points))
            )
        )

        dists = np.cumsum(
            [0] + [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        )
        sample_d = np.linspace(0, dists[-1], n_points)

        sampled = []
        j = 0
        for d in sample_d:
            while j < len(dists) - 2 and d > dists[j + 1]:
                j += 1
            ratio = (d - dists[j]) / (dists[j + 1] - dists[j] + 1e-8)
            interp = pts[j] + ratio * (pts[j + 1] - pts[j])
            sampled.append(tuple(interp.tolist()))

        # Apply Bezier curve smoothing for circular paths
        if smooth_circular_paths and is_circular_path(sampled):
            sampled = fit_bezier_curve(sampled, tolerance=curve_smoothing_tolerance)

        sampled_paths.append(sampled)

    # Step 4: sort paths (optional)
    if sort_paths:
        sampled_paths = sort_paths_human_like_final(sampled_paths)

    return sampled_paths
