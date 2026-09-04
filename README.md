# VParticleLife

A real-time 2D particle-life simulation written in C++20.

Particles of different types interact through a configurable interaction matrix. Simple local interaction rules can produce clustering, separation, oscillation, collective motion, and other emergent structures.

VParticleLife provides CPU and GPU simulation backends and is designed to run on both Windows and Linux.

## Features

* Multiple particle types with independent:

  * population
  * mass
  * size
  * color
* Configurable interaction matrix between particle types
* Optional symmetric interaction matrix
* Short-range core repulsion to prevent particle collapse
* Local velocity damping for dense clusters
* Global damping
* Periodic and reflective world boundaries
* Uniform-grid spatial partitioning for neighbor search
* Randomized:

  * particle positions
  * velocities
  * masses
  * interaction matrices
* Adjustable simulation timestep and speed
* Real-time camera zoom and pan
* Optional spatial grid visualization
* Real-time performance statistics
* Configurable fluorescent particle glow / bloom effect
* Dear ImGui control panel for live parameter editing

## Compute Backends

VParticleLife currently provides three simulation backends.

### CPU / OpenMP

The CPU backend uses a uniform spatial grid and OpenMP parallelization.

It is always available and also serves as the fallback backend when GPU compute is unavailable.

### OpenGL Compute

The OpenGL backend uses compute shaders and shader storage buffer objects.

It provides a cross-vendor GPU path and requires OpenGL 4.3 or newer.

If compute shaders are unavailable, rendering can still run with OpenGL 3.3 while the simulation falls back to the CPU backend.

### CUDA

An optional CUDA backend is available for NVIDIA GPUs.

It uses CUDA/OpenGL interoperability so that simulation buffers can be rendered directly without copying the complete particle state back to the CPU every frame.

The CUDA backend includes GPU-side spatial reordering, prefix scans, and optimized neighbor processing.

If CUDA is unavailable or initialization fails, VParticleLife automatically falls back to OpenGL Compute or CPU/OpenMP.

## Rendering

Particle rendering is handled with OpenGL.

The renderer supports:

* GPU-side particle buffers
* per-type particle sizes and colors
* additive particle accumulation
* order-independent rendering
* half-resolution bloom
* configurable glow strength, radius, density response, and exposure

Shaders and fonts are stored as external resources during development and can be embedded directly into release builds.

## Requirements

### Required

* C++20 compiler
* [Xmake](https://xmake.io/)
* OpenGL
* GLFW
* GLAD
* Dear ImGui
* OpenMP-capable compiler/runtime

GLFW, GLAD, ImGui, and OpenMP are managed by Xmake.

### Optional

For the CUDA backend:

* NVIDIA GPU
* NVIDIA driver
* CUDA Toolkit

Without CUDA, the project can still use OpenGL Compute or CPU/OpenMP.

## Supported Platforms

VParticleLife is currently developed and tested on:

* Windows
* Linux

The project is primarily intended for desktop systems with modern OpenGL support.

## Build

### Native Build

The default target is `vpl_native`.

It is intended for running VParticleLife on the current machine and enables aggressive native CPU optimizations.

Configure and build:

```bash
xmake f -c
xmake
```

Run:

```bash
xmake run vpl_native
```

CUDA is enabled by default when the CUDA backend option is enabled.

To explicitly disable CUDA:

```bash
xmake f -c --cuda_backend=n
xmake
```

To explicitly enable CUDA:

```bash
xmake f -c --cuda_backend=y
xmake
```

On systems where CUDA requires a specific host compiler, it can be selected through Xmake, for example:

```bash
xmake f -c --cuda_backend=y --cu-ccbin=/usr/bin/g++-15
```

### Debug Build

The debug target disables optimization and retains debug information.

```bash
xmake b vpl_debug
```

Run:

```bash
xmake run vpl_debug
```

Shaders and fonts remain external files in this build, which makes it suitable for development.

### Release Build

The release target is intended for distribution.

```bash
xmake b vpl_release
```

The resulting executable is placed in:

```text
dist/
```

Release builds:

* use aggressive optimization
* enable LTO
* embed shaders and fonts into the executable
* avoid native CPU architecture requirements
* include CUDA code for multiple NVIDIA GPU generations when CUDA is enabled

On Windows, the release target is built as a GUI application without a console window.

## Build Targets

| Target        | Purpose                      | Resources | CPU optimization | CUDA architecture        |
| ------------- | ---------------------------- | --------- | ---------------- | ------------------------ |
| `vpl_debug`   | Debugging                    | External  | None             | Current GPU              |
| `vpl_native`  | Local high-performance build | External  | Native machine   | Current GPU              |
| `vpl_release` | Distribution                 | Embedded  | Generic x86-64   | Multiple GPU generations |

## Project Structure

```text
VParticleLife/
├── src/
│   ├── main.cpp
│   ├── simulation.hpp
│   ├── opengl_compute.hpp
│   ├── cuda_compute.hpp
│   ├── cuda_compute.cu
│   └── embedded_resources.hpp
├── resources/
│   ├── shaders/
│   │   ├── particle.vert
│   │   ├── particle.frag
│   │   ├── particle_composite.vert
│   │   ├── particle_composite.frag
│   │   ├── particle_glow_extract.frag
│   │   ├── particle_glow_blur.frag
│   │   └── simulation.comp
│   └── fonts/
├── xmake.lua
├── README.md
└── LICENSE
```

## Simulation Model

Each particle belongs to a particle type.

For a particle of type `i`, the interaction with a neighboring particle of type `j` is controlled by the matrix element

```text
M[i][j]
```

The matrix does not need to be symmetric, so in general:

```text
M[i][j] != M[j][i]
```

This allows non-reciprocal interactions between particle types.

Particles only interact within a finite interaction radius.

At very short distances, a repulsive core force prevents particles from collapsing into the same position. Outside the core region, the interaction matrix determines the attractive or repulsive force between particle types.

The simulation also includes global damping and optional local velocity relaxation inside dense particle clusters.

Together, these simple local rules can generate complex self-organized behavior.

## Controls

Most simulation parameters can be modified at runtime through the Dear ImGui control panel, including:

* random seed and initialization ranges
* particle type count
* particles per type
* particle mass
* particle size
* particle color
* interaction matrix
* symmetric matrix mode
* core radius and strength
* global and local damping
* world dimensions
* boundary conditions
* timestep and simulation speed
* compute backend
* CPU thread count
* OpenGL compute work-group size
* CUDA device
* camera and rendering settings
* glow parameters
* VSync and FPS limit

The simulation can be paused, stepped manually, randomized, and reinitialized without restarting the application.

## Status

VParticleLife is an experimental project under active development.

Current development focuses on:

* simulation performance
* numerical stability
* dense particle aggregation behavior
* CPU/GPU backend optimization
* cross-platform support
* richer emergent particle behavior

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.
