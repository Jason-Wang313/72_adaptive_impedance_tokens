# Novelty Boundary Map

## Crowded Territory

- Adaptive impedance control for force tracking.
- Admittance-impedance switching.
- Robust and model-predictive contact controllers.
- Learned gain adaptation from contact observations.
- Hand-coded contact mode schedules.

## Tested Boundary

The tested boundary was whether discrete impedance tokens plus contact-outcome updates create a useful action-level representation beyond continuous learned gain adaptation.

## What Falsified The Boundary

The learned gain regressor reached 0.929 combined-stress success while the token policy reached 0.488. The `token_no_memory` ablation also beat the full token method. These two facts falsify the present novelty story.

## Remaining Possible Boundary

A future project could still explore learned token discovery or sequence-level token planning, but that is not demonstrated here.
