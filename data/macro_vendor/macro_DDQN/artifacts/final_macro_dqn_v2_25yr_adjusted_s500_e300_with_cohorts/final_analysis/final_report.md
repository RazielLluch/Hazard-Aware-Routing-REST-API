# Final Macro-DQN Evaluation Summary

This report is generated from the final artifact folder.

## Evaluation Lenses

- `random`: broad random 5-delivery scenarios across RI1-RI5.
- `hazard_opportunity`: RI2-RI5 scenarios where hazard-aware greedy routing can reduce common risk by at least 10.
- `risk_time_tradeoff`: stricter RI2-RI5 scenarios where the safer greedy route reduces common risk by at least 10 and costs at least 5 extra minutes.

## Key Profile Result

### random

Safe minus fast common risk: `-3.605073`
Safe minus fast time: `11.914656` minutes
Safe lower-risk share: `79.0%`

### hazard_opportunity

Safe minus fast common risk: `-3.643568`
Safe minus fast time: `12.081812` minutes
Safe lower-risk share: `78.083333%`

### risk_time_tradeoff

Safe minus fast common risk: `-6.07615`
Safe minus fast time: `18.871956` minutes
Safe lower-risk share: `80.0%`

## Key Baseline Result

### random

Safe Macro-DQN vs Greedy-HazardAware common risk delta: `-7.267188`
Safe Macro-DQN vs Greedy-HazardAware time delta: `-7.905249` minutes

### hazard_opportunity

Safe Macro-DQN vs Greedy-HazardAware common risk delta: `-7.153965`
Safe Macro-DQN vs Greedy-HazardAware time delta: `-8.963333` minutes

### risk_time_tradeoff

Safe Macro-DQN vs Greedy-HazardAware common risk delta: `-20.311286`
Safe Macro-DQN vs Greedy-HazardAware time delta: `-30.281899` minutes

## Profile Means

### random

- `fast`: time `209.514349`, common risk `206.392587`, distance `25.005315` km
- `balanced`: time `214.701419`, common risk `203.137338`, distance `27.274429` km
- `safe`: time `221.429006`, common risk `202.787514`, distance `30.063304` km

### hazard_opportunity

- `fast`: time `218.086819`, common risk `215.693142`, distance `26.019915` km
- `balanced`: time `223.135231`, common risk `212.394487`, distance `28.325341` km
- `safe`: time `230.16863`, common risk `212.049575`, distance `31.232121` km

### risk_time_tradeoff

- `fast`: time `225.805059`, common risk `239.306639`, distance `24.33805` km
- `balanced`: time `236.975403`, common risk `233.946183`, distance `28.850015` km
- `safe`: time `244.677015`, common risk `233.230489`, distance `32.080701` km
