# VParticleLife

A real-time 2D particle-life simulation written in C++17.

Particles of different types interact through a configurable interaction matrix. Simple local rules can produce clustering, separation, oscillation, and other emergent structures.

## Features

- Multiple particle types with independent colors, masses, and populations
- Configurable interaction matrix between particle types
- Short-range core repulsion and local damping for stability
- Periodic or collision-style world boundaries
- Uniform-grid neighbor search for local interactions
- CPU simulation backend with OpenMP support
- GPU acceleration with OpenGL compute shaders and CUDA
- Real-time OpenGL particle rendering
- Dear ImGui control panel for live parameter editing
- Camera zoom and pan
- Adjustable particle size, world size, timestep, damping, grid display, and rendering effects
- Randomized initial conditions and interaction parameters

## Requirements

- C++17 compiler
- [xmake](https://xmake.io/)
- OpenGL
- GLFW
- GLAD
- Dear ImGui
- OpenMP-capable compiler for the CPU backend
- CUDA Toolkit for the CUDA backend (optional)

The project is primarily developed and tested on Windows with MSVC and NVIDIA GPUs.

## Build

Clone the repository and build with xmake:

```bash
xmake
```

For the optimized release target used during development:

```bash
xmake b vpl_release
```

Then run the generated executable from the build directory, or use the corresponding xmake run target if configured.

## Project Structure

```text
ParticleLife/
├── src/
│   └── main.cpp
├── resources/
│   ├── shaders/
│   │   ├── particle.vert
│   │   └── particle.frag
│   └── ...
├── xmake.lua
└── README.md
```

## Model

Each particle belongs to a type. The interaction between type `i` and type `j` is controlled by an element of the interaction matrix.

At very short distances, particles repel each other to prevent collapse. Inside the interaction radius, the matrix determines whether particles attract or repel. Together with damping and local neighbor interactions, these simple rules can generate complex self-organized behavior.

The interaction matrix does not have to be symmetric, so the force exerted by type `i` on type `j` may differ from the reverse interaction.

## Status

ParticleLife is an experimental project under active development. The current focus is simulation performance, numerical stability, GPU acceleration, and richer emergent behavior.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
