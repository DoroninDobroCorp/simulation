# Architecture

## Simulation Engine

- Physics calculations run at fixed timestep
- Rendering decoupled from simulation
- Spatial partitioning for collision detection

## Performance

- Web Workers for parallel computation
- RequestAnimationFrame for smooth rendering