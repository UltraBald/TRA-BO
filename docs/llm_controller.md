# LLM acquisition controller

The controller chooses one acquisition function before the numerical optimizer scores the remaining candidate pool. It never receives future-window losses or candidate-specific acquisition scores.

## Input

`BOState.as_prompt_payload()` serializes the following fields:

- `tra`, `tra_valid`: the current Temporal Rank Agreement and whether both rankings were non-constant.
- `evaluations`, `initial_evaluations`, `remaining_budget`, `total_budget`, `progress`: budget state.
- `dimension`: latent search-space dimension.
- `incumbent`, `incumbent_history`: current and historical best calibrated targets.
- `target_mean`, `target_std`, `target_range`: calibrated-target diagnostics.
- `latest_min_distance`: distance from the newest evaluated point to earlier observations in the normalized space.
- `gp_uncertainty_mean`, `gp_uncertainty_max`: predictive uncertainty over the remaining candidate pool.
- `gp_lengthscale_min`, `gp_lengthscale_mean`, `gp_lengthscale_max`, `gp_output_scale`: GP posterior diagnostics.
- `previous_acquisition`, `acquisition_history`, `improvement_history`: sequential feedback.
- `non_improvement_streak`, `action_usage`, `action_successes`: summaries derived from the history.

The full system prompt is stored in `src/tra_bo/prompts/acquisition_controller.txt`.

## Output

The model must return one JSON object conforming to `schemas/acquisition_response.schema.json`, for example:

```json
{"acquisition": "EI"}
```

The parser accepts only PI, EI, UCB, or TS. An invalid response or request failure uses the configured fallback acquisition, which defaults to EI. Every call records the model identifier, input state, raw response, selected acquisition, fallback flag, error string, and latency. Credentials are never included.
