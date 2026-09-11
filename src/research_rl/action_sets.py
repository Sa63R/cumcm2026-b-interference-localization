"""Explicit candidate-set identity, independent of tensor feature dimension."""

ANYPOINT_SCHEMAS = ("anypoint_current", "anypoint_targets")
ACTION_SCHEMA_NAMES = ("base", "axis_quantiles", "range_probes", *ANYPOINT_SCHEMAS)


def action_schema(name="base"):
    if name == "base":
        return {"version": 1, "name": "base"}
    if name == "axis_quantiles":
        return {"version": 1, "name": "axis_quantiles", "extends": "v3-base-probes",
                "point_rule": "diameter-axis-q25-q375-q625-q75-width-over8",
                "max_extra_per_source": 14, "option_id": 6,
                "minimum_reception_margin_m": 1e-7}
    if name in ANYPOINT_SCHEMAS:
        return {"version": 1, "name": name, "extends": "v3-base-actions",
                "new_action": "measure-undetected-channel-at-observed-state-point",
                "max_extra_measurements": 32,
                "max_target_points": 4 if name == "anypoint_targets" else 0,
                "target_rule": "existing-probe-or-clear-points-nearest-distance-then-x-y",
                "point_deduplication": "exact-Position-no-rounding",
                "measurement_deduplication": "exact-accepted-position-channel",
                "unknown_rule": "1..20-minus-known-or-seven-site-certified-absent-unless-16-known",
                "cover_certificate": "only-actual-measurements-at-original-seven-points",
                "features": {"width": 60, "kind": "single-channel-cover",
                             "option_feature_index": 18, "option_divisor": 6,
                             "current_option_id": 7, "target_option_id": 8,
                             "point_pending": "original-cover-ledger-only-zero-elsewhere",
                             "scan_focus": "original-cover-focus-unchanged"},
                "network": "mlp-flat-only"}
    if name == "range_probes":
        return {"version": 1, "name": "range_probes", "extends": "v3-base-probes",
                "point_rule": "current-to-enclosing-center-midpoint-first-bearing-perpendicular",
                "offsets_m": [-150.0, -50.0, 50.0, 150.0],
                "max_extra_per_source": 4, "option_id": 7,
                "minimum_reception_margin_m": 1e-7}
    raise ValueError("unknown probe candidate family")


def validate_action_schema(spec):
    if spec is None:
        return action_schema()
    if not isinstance(spec, dict) or spec != action_schema(spec.get("name")):
        raise ValueError("unknown action schema metadata/version")
    return dict(spec)


def action_schema_from_args(args):
    return action_schema(getattr(args, "probe_candidates", "base"))


def checkpoint_action_schema(payload):
    spec = payload.get("action_schema")
    args = payload.get("args", {})
    if spec is None and args.get("probe_candidates", "base") != "base":
        raise ValueError("extended checkpoint is missing action schema metadata")
    result = validate_action_schema(spec)
    if "probe_candidates" in args and result != action_schema(args["probe_candidates"]):
        raise ValueError("action schema contradicts checkpoint arguments")
    return result


def controller_for(version, schema):
    schema = validate_action_schema(schema)
    if version == "v4":
        if schema["name"] != "base":
            raise ValueError("v4 route-debt ablation requires the unchanged base action set")
        from .route_debt import RouteDebtRLSearch
        return RouteDebtRLSearch
    if schema["name"] != "base":
        if version != "v3":
            raise ValueError("candidate extension requires v3 feature semantics")
        if schema["name"] in ANYPOINT_SCHEMAS:
            from .anypoint_scan import AnyPointCurrentRLSearch, AnyPointTargetsRLSearch
            return (AnyPointCurrentRLSearch if schema["name"] == "anypoint_current"
                    else AnyPointTargetsRLSearch)
        if schema["name"] == "range_probes":
            from .range_probes import RangeProbeRLSearch
            return RangeProbeRLSearch
        from .axis_probes import AxisProbeRLSearch
        return AxisProbeRLSearch
    if version == "v3":
        from .joint_scan import JointScanRLSearch
        return JointScanRLSearch
    from .controller import DeepRLSearch
    return DeepRLSearch
