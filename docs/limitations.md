# Limitations and future work

- **No verified shipment-delay outcome:** AIS observations describe vessel broadcasts, not shipment delivery commitments or lateness.
- **No public observations acquired:** the included reproducible development path is synthetic. It cannot support operational, geographic, or maritime-performance claims.
- **Coarse geography:** fixed bounding and candidate-anchorage geofences require validation with authoritative port boundaries and anchorage areas.
- **Proxy event:** the 25-vessel threshold is transparent but not an independently validated congestion definition.
- **AIS observation bias:** transmission gaps, reception coverage, duplicate messages, identity quality, and reporting frequency affect counts and dwell proxies.
- **Weak generalization evidence:** a single short temporal series is inadequate for claims about other ports, seasons, or operational conditions.
- **Calibration and reliability:** sample-based calibration and range checks do not ensure correctness under distribution shift; categories are diagnostics, not guarantees.
- **Explanations:** feature importance is not causality; local one-feature substitutions are sensitivity checks, not additive causal contributions.
- **Operational action:** recommendations are informational and must be reviewed by qualified staff; the system does not reroute vessels or make decisions.

Future work should acquire and archive licensed real AIS samples, validate geofences and anchorage-event labels with port experts, build leakage-safe vessel episodes, compare multiple seasons and ports, monitor distribution drift, and obtain verified shipment schedules/outcomes before any claim about shipment delays.
