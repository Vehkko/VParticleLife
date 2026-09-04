#include "cuda_compute.hpp"

#ifdef PARTICLELIFE_ENABLE_CUDA

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int MAX_TYPES = 16;
constexpr int TIMER_SLOTS = 4;
constexpr int DEFAULT_PARTICLE_THREADS = 256;
constexpr int DEFAULT_FORCE_THREADS = 256;

__constant__ float cInteraction[MAX_TYPES * MAX_TYPES];
__constant__ float cInvMass[MAX_TYPES];

struct DeviceParams {
    std::uint32_t particleCount;
    int gridCountX;
    int gridCountY;
    int neighborRangeX;
    int neighborRangeY;
    int interactionWidth;

    float worldWidth;
    float worldHeight;
    float halfWidth;
    float halfHeight;
    float invCellWidth;
    float invCellHeight;
    float restitution;

    float coreRadius2;
    float coreStrength;
    float interactionSpanInv;
    float localDampingDt2;
    float globalDampingFactor;
};

inline bool cuda_ok(cudaError_t error, const char* what) {
    if (error == cudaSuccess)
        return true;
    std::cerr << "CUDA error in " << what << ": "
              << cudaGetErrorString(error) << '\n';
    return false;
}

inline int round_warp_multiple(int value, int maxValue) {
    value = std::max(32, std::min(value, maxValue));
    value = (value / 32) * 32;
    return std::max(32, value);
}

__device__ __forceinline__ int clamp_cell(int value, int count) {
    return value < 0 ? 0 : (value >= count ? count - 1 : value);
}

__device__ __forceinline__ std::uint32_t cell_index(float x, float y,
                                                     const DeviceParams& p) {
    int cellX = static_cast<int>((x + p.halfWidth) * p.invCellWidth);
    int cellY = static_cast<int>((y + p.halfHeight) * p.invCellHeight);
    cellX = clamp_cell(cellX, p.gridCountX);
    cellY = clamp_cell(cellY, p.gridCountY);
    return static_cast<std::uint32_t>(cellY * p.gridCountX + cellX);
}

template <bool PERIODIC>
__device__ __forceinline__ void apply_boundary(float4& s,
                                                const DeviceParams& p) {
    if constexpr (PERIODIC) {
        if (s.x >= p.halfWidth)
            s.x -= p.worldWidth;
        else if (s.x < -p.halfWidth)
            s.x += p.worldWidth;

        if (s.y >= p.halfHeight)
            s.y -= p.worldHeight;
        else if (s.y < -p.halfHeight)
            s.y += p.worldHeight;
        return;
    }

    if (s.x > p.halfWidth) {
        s.x = 2.0f * p.halfWidth - s.x;
        s.z = -s.z * p.restitution;
    } else if (s.x < -p.halfWidth) {
        s.x = -2.0f * p.halfWidth - s.x;
        s.z = -s.z * p.restitution;
    }

    if (s.y > p.halfHeight) {
        s.y = 2.0f * p.halfHeight - s.y;
        s.w = -s.w * p.restitution;
    } else if (s.y < -p.halfHeight) {
        s.y = -2.0f * p.halfHeight - s.y;
        s.w = -s.w * p.restitution;
    }
}

__device__ __forceinline__ void periodic_delta(float& dx, float& dy,
                                                const DeviceParams& p) {
    if (dx > p.halfWidth)
        dx -= p.worldWidth;
    else if (dx < -p.halfWidth)
        dx += p.worldWidth;

    if (dy > p.halfHeight)
        dy -= p.worldHeight;
    else if (dy < -p.halfHeight)
        dy += p.worldHeight;
}

template <bool PERIODIC>
__global__ void kick_drift_count_kernel(float4* __restrict__ state,
                                        const float2* __restrict__ acceleration,
                                        std::uint32_t* __restrict__ cellCount,
                                        DeviceParams p,
                                        float halfDt, float dt) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.particleCount)
        return;

    float4 s = state[i];
    const float2 a = acceleration[i];
    s.z += a.x * halfDt;
    s.w += a.y * halfDt;
    s.x += s.z * dt;
    s.y += s.w * dt;
    apply_boundary<PERIODIC>(s, p);
    state[i] = s;

    atomicAdd(&cellCount[cell_index(s.x, s.y, p)], 1u);
}

__global__ void count_grid_kernel(const float4* __restrict__ state,
                                  std::uint32_t* __restrict__ cellCount,
                                  DeviceParams p) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.particleCount)
        return;
    const float4 s = state[i];
    atomicAdd(&cellCount[cell_index(s.x, s.y, p)], 1u);
}

__global__ void scatter_reorder_kernel(
    const float4* __restrict__ state,
    const std::uint32_t* __restrict__ type,
    float4* __restrict__ nextState,
    std::uint32_t* __restrict__ nextType,
    std::uint32_t* __restrict__ cellWrite,
    const std::uint32_t* __restrict__ cellOffset,
    DeviceParams p) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.particleCount)
        return;

    const float4 s = state[i];
    const std::uint32_t cell = cell_index(s.x, s.y, p);
    const std::uint32_t rank = atomicAdd(&cellWrite[cell], 1u);
    const std::uint32_t dst = cellOffset[cell] + rank;
    nextState[dst] = s;
    nextType[dst] = type[i];
}

struct ForceAccum {
    float ax = 0.0f;
    float ay = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float weight = 0.0f;
};

template <bool PERIODIC>
__device__ __forceinline__ void accumulate_pair(
    std::uint32_t i, std::uint32_t j,
    const float4& si, std::uint32_t typeI,
    const float4& sj, std::uint32_t typeJ,
    const DeviceParams& p, ForceAccum& out) {
    if (j == i)
        return;

    float dx = sj.x - si.x;
    float dy = sj.y - si.y;
    if constexpr (PERIODIC)
        periodic_delta(dx, dy, p);

    const float d2 = dx * dx + dy * dy;
    if (d2 > INTERACTION_RADIUS_SQUARE)
        return;

    float force;
    if (d2 < p.coreRadius2) {
        const float safeD2 = fmaxf(d2, MIN_DISTANCE_SQUARE);
        force = -p.coreStrength / safeD2;

        if (p.localDampingDt2 > 0.0f) {
            const float weight = 1.0f - d2 / p.coreRadius2;
            out.vx += weight * sj.z;
            out.vy += weight * sj.w;
            out.weight += weight;
        }
    } else {
        const float t = (d2 - p.coreRadius2) * p.interactionSpanInv;
        const float g = 4.0f * t * (1.0f - t);
        force = -cInteraction[typeI * p.interactionWidth + typeJ] * g;
    }

    out.ax += force * dx;
    out.ay += force * dy;
}

template <bool PERIODIC>
__device__ __forceinline__ void force_fallback_one(
    std::uint32_t i,
    const float4* __restrict__ state,
    const std::uint32_t* __restrict__ type,
    const std::uint32_t* __restrict__ cellOffset,
    DeviceParams p,
    ForceAccum& out) {
    const float4 si = state[i];
    const std::uint32_t typeI = type[i];
    const std::uint32_t ownCell = cell_index(si.x, si.y, p);
    const int cellX = static_cast<int>(ownCell % static_cast<std::uint32_t>(p.gridCountX));
    const int cellY = static_cast<int>(ownCell / static_cast<std::uint32_t>(p.gridCountX));

    for (int oy = -p.neighborRangeY; oy <= p.neighborRangeY; ++oy) {
        int ny = cellY + oy;
        if constexpr (PERIODIC) {
            if (ny < 0)
                ny += p.gridCountY;
            else if (ny >= p.gridCountY)
                ny -= p.gridCountY;
        } else if (ny < 0 || ny >= p.gridCountY) {
            continue;
        }

        for (int ox = -p.neighborRangeX; ox <= p.neighborRangeX; ++ox) {
            int nx = cellX + ox;
            if constexpr (PERIODIC) {
                if (nx < 0)
                    nx += p.gridCountX;
                else if (nx >= p.gridCountX)
                    nx -= p.gridCountX;
            } else if (nx < 0 || nx >= p.gridCountX) {
                continue;
            }

            const std::uint32_t cell =
                static_cast<std::uint32_t>(ny * p.gridCountX + nx);
            const std::uint32_t begin = cellOffset[cell];
            const std::uint32_t end = cellOffset[cell + 1u];
            for (std::uint32_t j = begin; j < end; ++j)
                accumulate_pair<PERIODIC>(i, j, si, typeI,
                                          state[j], type[j], p, out);
        }
    }
}

// Hybrid force kernel:
// - physical cell ordering guarantees contiguous blocks often belong to one cell
// - dense homogeneous blocks cooperatively stage neighbor tiles in shared memory
// - sparse/mixed blocks fall back to the ordinary contiguous global-memory walk
// This targets the expensive late-time dense clusters without making sparse scenes
// pay the cost of a cell-work queue or another scan.
template <bool PERIODIC>
__global__ void force_kernel(
    const float4* __restrict__ state,
    const std::uint32_t* __restrict__ type,
    float2* __restrict__ acceleration,
    float4* __restrict__ relaxation,
    const std::uint32_t* __restrict__ cellOffset,
    DeviceParams p) {
    extern __shared__ unsigned char sharedRaw[];
    float4* sharedState = reinterpret_cast<float4*>(sharedRaw);
    std::uint32_t* sharedType = reinterpret_cast<std::uint32_t*>(
        sharedState + blockDim.x);

    __shared__ std::uint32_t targetCellShared;
    __shared__ int homogeneousShared;

    const std::uint32_t blockBegin = blockIdx.x * blockDim.x;
    if (threadIdx.x == 0) {
        if (blockBegin >= p.particleCount) {
            homogeneousShared = 0;
            targetCellShared = 0;
        } else {
            const std::uint32_t blockEnd =
                min(blockBegin + static_cast<std::uint32_t>(blockDim.x),
                    p.particleCount) - 1u;
            const float4 first = state[blockBegin];
            const float4 last = state[blockEnd];
            const std::uint32_t firstCell = cell_index(first.x, first.y, p);
            const std::uint32_t lastCell = cell_index(last.x, last.y, p);
            homogeneousShared = firstCell == lastCell ? 1 : 0;
            targetCellShared = firstCell;
        }
    }
    __syncthreads();

    const std::uint32_t i = blockBegin + threadIdx.x;
    const bool active = i < p.particleCount;

    if (!homogeneousShared) {
        if (!active)
            return;

        ForceAccum out;
        force_fallback_one<PERIODIC>(i, state, type, cellOffset, p, out);
        const std::uint32_t typeI = type[i];
        const float invMass = cInvMass[typeI];
        acceleration[i] = make_float2(out.ax * invMass, out.ay * invMass);
        relaxation[i] = make_float4(out.vx, out.vy, out.weight, 0.0f);
        return;
    }

    float4 si{};
    std::uint32_t typeI = 0;
    ForceAccum out;
    if (active) {
        si = state[i];
        typeI = type[i];
    }

    const int cellX = static_cast<int>(
        targetCellShared % static_cast<std::uint32_t>(p.gridCountX));
    const int cellY = static_cast<int>(
        targetCellShared / static_cast<std::uint32_t>(p.gridCountX));

    for (int oy = -p.neighborRangeY; oy <= p.neighborRangeY; ++oy) {
        int ny = cellY + oy;
        if constexpr (PERIODIC) {
            if (ny < 0)
                ny += p.gridCountY;
            else if (ny >= p.gridCountY)
                ny -= p.gridCountY;
        } else if (ny < 0 || ny >= p.gridCountY) {
            continue;
        }

        for (int ox = -p.neighborRangeX; ox <= p.neighborRangeX; ++ox) {
            int nx = cellX + ox;
            if constexpr (PERIODIC) {
                if (nx < 0)
                    nx += p.gridCountX;
                else if (nx >= p.gridCountX)
                    nx -= p.gridCountX;
            } else if (nx < 0 || nx >= p.gridCountX) {
                continue;
            }

            const std::uint32_t cell =
                static_cast<std::uint32_t>(ny * p.gridCountX + nx);
            const std::uint32_t begin = cellOffset[cell];
            const std::uint32_t end = cellOffset[cell + 1u];

            for (std::uint32_t tile = begin; tile < end;
                 tile += static_cast<std::uint32_t>(blockDim.x)) {
                const std::uint32_t jLoad = tile + threadIdx.x;
                if (jLoad < end) {
                    sharedState[threadIdx.x] = state[jLoad];
                    sharedType[threadIdx.x] = type[jLoad];
                }
                __syncthreads();

                if (active) {
                    const std::uint32_t tileCount =
                        min(static_cast<std::uint32_t>(blockDim.x), end - tile);
                    #pragma unroll 4
                    for (std::uint32_t k = 0; k < tileCount; ++k) {
                        accumulate_pair<PERIODIC>(
                            i, tile + k, si, typeI,
                            sharedState[k], sharedType[k], p, out);
                    }
                }
                __syncthreads();
            }
        }
    }

    if (active) {
        const float invMass = cInvMass[typeI];
        acceleration[i] = make_float2(out.ax * invMass, out.ay * invMass);
        relaxation[i] = make_float4(out.vx, out.vy, out.weight, 0.0f);
    }
}

__device__ __forceinline__ float2 finish_velocity(
    float2 v, const float2 a, const float4 local,
    std::uint32_t typeI, const DeviceParams& p, float halfDt) {
    v.x += a.x * halfDt;
    v.y += a.y * halfDt;

    if (local.z > 0.0f && p.localDampingDt2 > 0.0f) {
        const float invWeight = 1.0f / local.z;
        const float meanVx = local.x * invWeight;
        const float meanVy = local.y * invWeight;
        const float exponent = -p.localDampingDt2 * local.z * cInvMass[typeI];
        const float blend = 0.5f * (1.0f - __expf(exponent));
        v.x += blend * (meanVx - v.x);
        v.y += blend * (meanVy - v.y);
    }

    v.x *= p.globalDampingFactor;
    v.y *= p.globalDampingFactor;
    return v;
}

__global__ void kick_damping_kernel(
    float4* __restrict__ state,
    const std::uint32_t* __restrict__ type,
    const float2* __restrict__ acceleration,
    const float4* __restrict__ relaxation,
    DeviceParams p, float halfDt) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.particleCount)
        return;

    float4 s = state[i];
    const float2 v = finish_velocity(make_float2(s.z, s.w), acceleration[i],
                                     relaxation[i], type[i], p, halfDt);
    s.z = v.x;
    s.w = v.y;
    state[i] = s;
}

template <bool PERIODIC>
__global__ void finish_begin_next_count_kernel(
    float4* __restrict__ state,
    const std::uint32_t* __restrict__ type,
    const float2* __restrict__ acceleration,
    const float4* __restrict__ relaxation,
    std::uint32_t* __restrict__ cellCount,
    DeviceParams p, float halfDt, float dt) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.particleCount)
        return;

    float4 s = state[i];
    float2 v = finish_velocity(make_float2(s.z, s.w), acceleration[i],
                               relaxation[i], type[i], p, halfDt);

    // First half-kick of the next Velocity-Verlet step uses the same current
    // acceleration, matching the OpenGL backend's fused stage.
    v.x += acceleration[i].x * halfDt;
    v.y += acceleration[i].y * halfDt;
    s.z = v.x;
    s.w = v.y;
    s.x += s.z * dt;
    s.y += s.w * dt;
    apply_boundary<PERIODIC>(s, p);
    state[i] = s;

    atomicAdd(&cellCount[cell_index(s.x, s.y, p)], 1u);
}

} // namespace

struct CudaComputeBackend::Impl {
    int deviceId = -1;
    std::string deviceName;
    bool failed = false;

    cudaDeviceProp deviceProp{};
    cudaStream_t stream = nullptr;

    // CUDA owns the authoritative simulation buffers. They are stable pointers,
    // which keeps the hot path independent from OpenGL mapping semantics.
    std::array<float4*, 2> state{};
    std::array<std::uint32_t*, 2> type{};
    int current = 0;

    float2* acceleration = nullptr;
    float4* relaxation = nullptr;
    std::uint32_t* cellCount = nullptr;   // cellCount + one zero sentinel
    std::uint32_t* cellOffset = nullptr;  // cellCount + 1

    void* scanTemp = nullptr;
    std::size_t scanTempBytes = 0;

    // A small render-only interop target. CUDA copies the final contiguous state
    // here once per rendered frame; simulation itself never waits on OpenGL.
    std::array<GLuint, 2> particleBuffer{};
    std::array<GLuint, 2> typeBuffer{};
    std::array<cudaGraphicsResource*, 2> particleResource{};
    std::array<cudaGraphicsResource*, 2> typeResource{};
    int renderIndex = 0;

    std::size_t particleCount = 0;
    std::size_t cellCountN = 0;
    int gridCountX = 1;
    int gridCountY = 1;
    float cellWidth = 1.0f;
    float cellHeight = 1.0f;
    int neighborRangeX = 1;
    int neighborRangeY = 1;

    int particleThreads = DEFAULT_PARTICLE_THREADS;
    int forceThreads = DEFAULT_FORCE_THREADS;

    std::array<cudaEvent_t, TIMER_SLOTS> timerStart{};
    std::array<cudaEvent_t, TIMER_SLOTS> timerStop{};
    std::array<bool, TIMER_SLOTS> timerPending{};
    std::array<int, TIMER_SLOTS> timerSteps{};
    std::size_t timerWrite = 0;
    double lastGpuMs = -1.0;
    int lastGpuSteps = 0;

    bool set_device(int id) {
        if (!cuda_ok(cudaSetDevice(id), "cudaSetDevice"))
            return false;
        deviceId = id;
        if (!cuda_ok(cudaGetDeviceProperties(&deviceProp, id),
                     "cudaGetDeviceProperties"))
            return false;
        deviceName = deviceProp.name;
        return true;
    }

    void choose_launch_sizes(bool periodic) {
        particleThreads = std::min(DEFAULT_PARTICLE_THREADS,
                                   deviceProp.maxThreadsPerBlock);
        particleThreads = round_warp_multiple(particleThreads,
                                              deviceProp.maxThreadsPerBlock);

        // Occupancy is not a complete performance model, but it is a robust
        // default for the register-heavy force kernel. Evaluate only practical
        // warp-aligned choices and prefer the smallest block on equal resident
        // thread count; smaller blocks handle cell boundaries more gracefully.
        constexpr int candidates[] = {128, 256, 512};
        int bestThreads = std::min(DEFAULT_FORCE_THREADS,
                                   deviceProp.maxThreadsPerBlock);
        int bestResidentThreads = -1;

        for (int candidate : candidates) {
            if (candidate > deviceProp.maxThreadsPerBlock)
                continue;
            const std::size_t sharedBytes =
                static_cast<std::size_t>(candidate) *
                (sizeof(float4) + sizeof(std::uint32_t));
            int activeBlocks = 0;
            cudaError_t status;
            if (periodic) {
                status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &activeBlocks, force_kernel<true>, candidate, sharedBytes);
            } else {
                status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &activeBlocks, force_kernel<false>, candidate, sharedBytes);
            }
            if (status != cudaSuccess) {
                cudaGetLastError();
                continue;
            }
            const int residentThreads = activeBlocks * candidate;
            if (residentThreads > bestResidentThreads) {
                bestResidentThreads = residentThreads;
                bestThreads = candidate;
            }
        }
        forceThreads = round_warp_multiple(bestThreads,
                                           deviceProp.maxThreadsPerBlock);
    }

    void free_device_arrays() {
        for (float4*& ptr : state) {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
        }
        for (std::uint32_t*& ptr : type) {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
        }
        if (acceleration) cudaFree(acceleration);
        if (relaxation) cudaFree(relaxation);
        if (cellCount) cudaFree(cellCount);
        if (cellOffset) cudaFree(cellOffset);
        if (scanTemp) cudaFree(scanTemp);
        acceleration = nullptr;
        relaxation = nullptr;
        cellCount = nullptr;
        cellOffset = nullptr;
        scanTemp = nullptr;
        scanTempBytes = 0;
        current = 0;
    }

    void destroy_interop_buffers() {
        for (int i = 0; i < 2; ++i) {
            if (particleResource[i])
                cudaGraphicsUnregisterResource(particleResource[i]);
            if (typeResource[i])
                cudaGraphicsUnregisterResource(typeResource[i]);
            particleResource[i] = nullptr;
            typeResource[i] = nullptr;

            if (particleBuffer[i])
                glDeleteBuffers(1, &particleBuffer[i]);
            if (typeBuffer[i])
                glDeleteBuffers(1, &typeBuffer[i]);
            particleBuffer[i] = 0;
            typeBuffer[i] = 0;
        }
        renderIndex = 0;
    }

    bool create_interop_buffers(std::size_t particles) {
        destroy_interop_buffers();

        for (int i = 0; i < 2; ++i) {
            glGenBuffers(1, &particleBuffer[i]);
            glBindBuffer(GL_ARRAY_BUFFER, particleBuffer[i]);
            glBufferData(GL_ARRAY_BUFFER,
                         static_cast<GLsizeiptr>(particles * sizeof(float4)),
                         nullptr, GL_STREAM_DRAW);

            glGenBuffers(1, &typeBuffer[i]);
            glBindBuffer(GL_ARRAY_BUFFER, typeBuffer[i]);
            glBufferData(GL_ARRAY_BUFFER,
                         static_cast<GLsizeiptr>(particles * sizeof(std::uint32_t)),
                         nullptr, GL_STREAM_DRAW);

            if (!cuda_ok(cudaGraphicsGLRegisterBuffer(
                             &particleResource[i], particleBuffer[i],
                             cudaGraphicsRegisterFlagsWriteDiscard),
                         "cudaGraphicsGLRegisterBuffer(particle)"))
                return false;

            if (!cuda_ok(cudaGraphicsGLRegisterBuffer(
                             &typeResource[i], typeBuffer[i],
                             cudaGraphicsRegisterFlagsWriteDiscard),
                         "cudaGraphicsGLRegisterBuffer(type)"))
                return false;
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return true;
    }

    bool allocate_device_arrays(const SimulationConfig& config,
                                std::size_t particles) {
        free_device_arrays();

        gridCountX = std::max(1, static_cast<int>(
            config.world.width / config.world.bucketSize));
        gridCountY = std::max(1, static_cast<int>(
            config.world.height / config.world.bucketSize));
        cellWidth = config.world.width / static_cast<float>(gridCountX);
        cellHeight = config.world.height / static_cast<float>(gridCountY);
        neighborRangeX = static_cast<int>(
            std::ceil(INTERACTION_RADIUS / cellWidth));
        neighborRangeY = static_cast<int>(
            std::ceil(INTERACTION_RADIUS / cellHeight));
        cellCountN = static_cast<std::size_t>(gridCountX) *
                     static_cast<std::size_t>(gridCountY);
        particleCount = particles;

        const std::size_t stateBytes = particles * sizeof(float4);
        const std::size_t typeBytes = particles * sizeof(std::uint32_t);
        const std::size_t cellBytes = (cellCountN + 1) * sizeof(std::uint32_t);

        bool ok = true;
        ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(&state[0]), stateBytes),
                           "cudaMalloc(state0)");
        ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(&state[1]), stateBytes),
                           "cudaMalloc(state1)");
        ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(&type[0]), typeBytes),
                           "cudaMalloc(type0)");
        ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(&type[1]), typeBytes),
                           "cudaMalloc(type1)");
        ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(&acceleration),
                                      particles * sizeof(float2)),
                           "cudaMalloc(acceleration)");
        ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(&relaxation),
                                      particles * sizeof(float4)),
                           "cudaMalloc(relaxation)");
        ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cellCount), cellBytes),
                           "cudaMalloc(cellCount)");
        ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cellOffset), cellBytes),
                           "cudaMalloc(cellOffset)");
        if (!ok)
            return false;

        // Ask CUB once for the exact temporary-storage requirement. The hot path
        // reuses this allocation for every grid rebuild.
        scanTempBytes = 0;
        if (!cuda_ok(cub::DeviceScan::ExclusiveSum(
                         nullptr, scanTempBytes,
                         cellCount, cellOffset,
                         static_cast<int>(cellCountN + 1), stream),
                     "cub::DeviceScan::ExclusiveSum(size)"))
            return false;
        if (scanTempBytes > 0 &&
            !cuda_ok(cudaMalloc(&scanTemp, scanTempBytes), "cudaMalloc(CUB scan temp)"))
            return false;

        choose_launch_sizes(config.world.boundary == BoundaryMode::Periodic);
        return true;
    }

    DeviceParams make_params(const SimulationConfig& config, float dt) const {
        DeviceParams p{};
        p.particleCount = static_cast<std::uint32_t>(particleCount);
        p.gridCountX = gridCountX;
        p.gridCountY = gridCountY;
        p.neighborRangeX = neighborRangeX;
        p.neighborRangeY = neighborRangeY;
        p.interactionWidth = static_cast<int>(config.interaction.width);
        p.worldWidth = config.world.width;
        p.worldHeight = config.world.height;
        p.halfWidth = 0.5f * config.world.width;
        p.halfHeight = 0.5f * config.world.height;
        p.invCellWidth = 1.0f / cellWidth;
        p.invCellHeight = 1.0f / cellHeight;
        p.restitution = config.world.restitution;
        p.coreRadius2 = config.physics.coreRadius * config.physics.coreRadius;
        p.coreStrength = config.physics.coreStrength;
        p.interactionSpanInv =
            1.0f / (INTERACTION_RADIUS_SQUARE - p.coreRadius2);
        p.localDampingDt2 = 2.0f * config.physics.localDamping * dt;
        p.globalDampingFactor = std::exp(-config.physics.damping * dt);
        return p;
    }

    bool upload_constants(const SimulationConfig& config) {
        if (config.particleTypes.size() > MAX_TYPES ||
            config.interaction.width > MAX_TYPES) {
            std::cerr << "CUDA backend supports at most " << MAX_TYPES
                      << " particle types.\n";
            return false;
        }

        std::array<float, MAX_TYPES * MAX_TYPES> interaction{};
        std::copy(config.interaction.data.begin(), config.interaction.data.end(),
                  interaction.begin());

        std::array<float, MAX_TYPES> invMass{};
        for (std::size_t i = 0; i < config.particleTypes.size(); ++i)
            invMass[i] = 1.0f / config.particleTypes[i].mass;

        return cuda_ok(cudaMemcpyToSymbol(cInteraction, interaction.data(),
                                          sizeof(interaction)),
                       "cudaMemcpyToSymbol(interaction)") &&
               cuda_ok(cudaMemcpyToSymbol(cInvMass, invMass.data(),
                                          sizeof(invMass)),
                       "cudaMemcpyToSymbol(invMass)");
    }

    int particle_blocks() const {
        return static_cast<int>((particleCount + particleThreads - 1) /
                                particleThreads);
    }

    int force_blocks() const {
        return static_cast<int>((particleCount + forceThreads - 1) /
                                forceThreads);
    }

    std::size_t force_shared_bytes() const {
        return static_cast<std::size_t>(forceThreads) *
               (sizeof(float4) + sizeof(std::uint32_t));
    }

    bool clear_counts() {
        return cuda_ok(cudaMemsetAsync(cellCount, 0,
                                       (cellCountN + 1) * sizeof(std::uint32_t),
                                       stream),
                       "cudaMemsetAsync(cellCount)");
    }

    bool scan_counts() {
        return cuda_ok(cub::DeviceScan::ExclusiveSum(
                           scanTemp, scanTempBytes,
                           cellCount, cellOffset,
                           static_cast<int>(cellCountN + 1), stream),
                       "cub::DeviceScan::ExclusiveSum");
    }

    bool clear_write_counters() {
        // cellCount is dead after the scan, so reuse it as per-cell scatter cursor.
        return cuda_ok(cudaMemsetAsync(cellCount, 0,
                                       cellCountN * sizeof(std::uint32_t), stream),
                       "cudaMemsetAsync(cellWrite)");
    }

    template <bool PERIODIC>
    void launch_force(float4* s, std::uint32_t* t, const DeviceParams& p) {
        force_kernel<PERIODIC><<<force_blocks(), forceThreads,
                                 force_shared_bytes(), stream>>>(
            s, t, acceleration, relaxation, cellOffset, p);
    }

    template <bool PERIODIC>
    bool initialize_acceleration_impl(const SimulationConfig& config) {
        if (particleCount == 0)
            return true;

        const DeviceParams p = make_params(config, config.time.dt);
        if (!clear_counts())
            return false;
        count_grid_kernel<<<particle_blocks(), particleThreads, 0, stream>>>(
            state[current], cellCount, p);
        if (!scan_counts() || !clear_write_counters())
            return false;

        const int next = 1 - current;
        scatter_reorder_kernel<<<particle_blocks(), particleThreads, 0, stream>>>(
            state[current], type[current], state[next], type[next],
            cellCount, cellOffset, p);
        launch_force<PERIODIC>(state[next], type[next], p);
        current = next;

        if (!cuda_ok(cudaGetLastError(), "initialize_acceleration kernels"))
            return false;
        return cuda_ok(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize(initialize_acceleration)");
    }

    bool initialize_acceleration(const SimulationConfig& config) {
        return config.world.boundary == BoundaryMode::Periodic
            ? initialize_acceleration_impl<true>(config)
            : initialize_acceleration_impl<false>(config);
    }

    bool sync_render_buffers() {
        if (!particleResource[0] || !typeResource[0] || particleCount == 0)
            return true;

        // Write into the buffer not used by the previous frame. This avoids
        // forcing CUDA to wait on the immediately preceding OpenGL draw when
        // the graphics queue is still consuming that buffer.
        const int writeIndex = 1 - renderIndex;
        cudaGraphicsResource* resources[] = {
            particleResource[writeIndex], typeResource[writeIndex]};
        if (!cuda_ok(cudaGraphicsMapResources(2, resources, stream),
                     "cudaGraphicsMapResources(render)"))
            return false;

        float4* glState = nullptr;
        std::uint32_t* glType = nullptr;
        std::size_t stateBytes = 0;
        std::size_t typeBytes = 0;
        bool ok = cuda_ok(cudaGraphicsResourceGetMappedPointer(
                              reinterpret_cast<void**>(&glState), &stateBytes,
                              particleResource[writeIndex]),
                          "cudaGraphicsResourceGetMappedPointer(particle)") &&
                  cuda_ok(cudaGraphicsResourceGetMappedPointer(
                              reinterpret_cast<void**>(&glType), &typeBytes,
                              typeResource[writeIndex]),
                          "cudaGraphicsResourceGetMappedPointer(type)");

        const std::size_t requiredState = particleCount * sizeof(float4);
        const std::size_t requiredType = particleCount * sizeof(std::uint32_t);
        if (ok && (stateBytes < requiredState || typeBytes < requiredType)) {
            std::cerr << "CUDA/OpenGL render buffer is smaller than expected.\n";
            ok = false;
        }

        if (ok) {
            ok = cuda_ok(cudaMemcpyAsync(glState, state[current], requiredState,
                                         cudaMemcpyDeviceToDevice, stream),
                         "cudaMemcpyAsync(render state)") &&
                 cuda_ok(cudaMemcpyAsync(glType, type[current], requiredType,
                                         cudaMemcpyDeviceToDevice, stream),
                         "cudaMemcpyAsync(render type)");
        }

        const bool unmapOk = cuda_ok(cudaGraphicsUnmapResources(2, resources, stream),
                                     "cudaGraphicsUnmapResources(render)");
        if (ok && unmapOk)
            renderIndex = writeIndex;
        return ok && unmapOk;
    }

    void poll_timers() {
        for (int i = 0; i < TIMER_SLOTS; ++i) {
            if (!timerPending[static_cast<std::size_t>(i)])
                continue;
            const cudaError_t status =
                cudaEventQuery(timerStop[static_cast<std::size_t>(i)]);
            if (status == cudaErrorNotReady)
                continue;
            if (status != cudaSuccess) {
                cudaGetLastError();
                timerPending[static_cast<std::size_t>(i)] = false;
                continue;
            }
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms,
                                     timerStart[static_cast<std::size_t>(i)],
                                     timerStop[static_cast<std::size_t>(i)]) == cudaSuccess) {
                lastGpuMs = ms;
                lastGpuSteps = timerSteps[static_cast<std::size_t>(i)];
            }
            timerPending[static_cast<std::size_t>(i)] = false;
        }
    }
};

CudaComputeBackend::CudaComputeBackend() : impl_(new Impl) {}

CudaComputeBackend::~CudaComputeBackend() {
    destroy();
    delete impl_;
    impl_ = nullptr;
}

std::vector<CudaDeviceInfo> CudaComputeBackend::enumerate_devices() {
    std::vector<CudaDeviceInfo> result;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return result;
    }

    std::vector<int> glDevices(static_cast<std::size_t>(std::max(1, count)), -1);
    unsigned int glCount = 0;
    const cudaError_t glStatus = cudaGLGetDevices(
        &glCount, glDevices.data(), static_cast<unsigned int>(glDevices.size()),
        cudaGLDeviceListAll);
    if (glStatus != cudaSuccess) {
        cudaGetLastError();
        glCount = 0;
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }

        CudaDeviceInfo info;
        info.id = i;
        info.name = prop.name;
        info.totalMemory = static_cast<std::uint64_t>(prop.totalGlobalMem);
        info.major = prop.major;
        info.minor = prop.minor;
        info.multiprocessorCount = prop.multiProcessorCount;
        info.warpSize = prop.warpSize;
        info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
        info.openGLCompatible = std::find(glDevices.begin(),
                                          glDevices.begin() + glCount,
                                          i) != glDevices.begin() + glCount;
        result.push_back(std::move(info));
    }
    return result;
}

bool CudaComputeBackend::initialize(const Simulation& simulation,
                                    int requestedDevice) {
    if (!impl_)
        return false;
    destroy();

    const std::vector<CudaDeviceInfo> devices = enumerate_devices();
    int selected = -1;
    if (requestedDevice >= 0) {
        for (const CudaDeviceInfo& device : devices)
            if (device.id == requestedDevice && device.openGLCompatible)
                selected = device.id;
    }
    if (selected < 0) {
        for (const CudaDeviceInfo& device : devices) {
            if (device.openGLCompatible) {
                selected = device.id;
                break;
            }
        }
    }

    if (selected < 0) {
        std::cerr << "CUDA backend: no CUDA device compatible with the current "
                     "OpenGL context was found.\n";
        return false;
    }

    if (!impl_->set_device(selected))
        return false;

    if (!cuda_ok(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags")) {
        destroy();
        return false;
    }

    for (int i = 0; i < TIMER_SLOTS; ++i) {
        if (!cuda_ok(cudaEventCreate(&impl_->timerStart[static_cast<std::size_t>(i)]),
                     "cudaEventCreate(start)") ||
            !cuda_ok(cudaEventCreate(&impl_->timerStop[static_cast<std::size_t>(i)]),
                     "cudaEventCreate(stop)")) {
            destroy();
            return false;
        }
    }

    rebuild(simulation);
    return valid();
}

void CudaComputeBackend::destroy() {
    if (!impl_)
        return;

    if (impl_->deviceId >= 0)
        cudaSetDevice(impl_->deviceId);

    impl_->destroy_interop_buffers();
    impl_->free_device_arrays();

    for (int i = 0; i < TIMER_SLOTS; ++i) {
        cudaEvent_t& start = impl_->timerStart[static_cast<std::size_t>(i)];
        cudaEvent_t& stop = impl_->timerStop[static_cast<std::size_t>(i)];
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        start = nullptr;
        stop = nullptr;
    }

    if (impl_->stream)
        cudaStreamDestroy(impl_->stream);
    impl_->stream = nullptr;

    impl_->timerPending.fill(false);
    impl_->deviceId = -1;
    impl_->deviceName.clear();
    impl_->particleCount = 0;
    impl_->cellCountN = 0;
    impl_->failed = false;
    impl_->lastGpuMs = -1.0;
    impl_->lastGpuSteps = 0;
    impl_->particleThreads = DEFAULT_PARTICLE_THREADS;
    impl_->forceThreads = DEFAULT_FORCE_THREADS;
}

bool CudaComputeBackend::valid() const noexcept {
    return impl_ && impl_->deviceId >= 0 && !impl_->failed &&
           impl_->state[impl_->current] && impl_->particleResource[0] &&
           impl_->typeResource[0];
}

int CudaComputeBackend::device_id() const noexcept {
    return impl_ ? impl_->deviceId : -1;
}

const char* CudaComputeBackend::device_name() const noexcept {
    return impl_ && !impl_->deviceName.empty()
        ? impl_->deviceName.c_str() : "Unavailable";
}

GLuint CudaComputeBackend::particle_buffer() const noexcept {
    return impl_ ? impl_->particleBuffer[impl_->renderIndex] : 0;
}

GLuint CudaComputeBackend::type_buffer() const noexcept {
    return impl_ ? impl_->typeBuffer[impl_->renderIndex] : 0;
}

double CudaComputeBackend::last_gpu_ms() const noexcept {
    return impl_ ? impl_->lastGpuMs : -1.0;
}

int CudaComputeBackend::last_gpu_steps() const noexcept {
    return impl_ ? impl_->lastGpuSteps : 0;
}

int CudaComputeBackend::particle_threads() const noexcept {
    return impl_ ? impl_->particleThreads : 0;
}

int CudaComputeBackend::force_threads() const noexcept {
    return impl_ ? impl_->forceThreads : 0;
}

void CudaComputeBackend::rebuild(const Simulation& simulation) {
    if (!impl_ || impl_->deviceId < 0)
        return;

    const std::size_t count = simulation.particles().particle_count();
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "CUDA backend: particle count exceeds uint32 range.\n";
        impl_->failed = true;
        return;
    }

    if (!impl_->create_interop_buffers(count) ||
        !impl_->allocate_device_arrays(simulation.config(), count) ||
        !impl_->upload_constants(simulation.config())) {
        impl_->failed = true;
        return;
    }

    upload_state(simulation);
}

void CudaComputeBackend::upload_state(const Simulation& simulation) {
    if (!impl_ || impl_->deviceId < 0 || impl_->failed)
        return;
    if (simulation.particles().particle_count() != impl_->particleCount) {
        rebuild(simulation);
        return;
    }
    if (!impl_->upload_constants(simulation.config())) {
        impl_->failed = true;
        return;
    }

    std::vector<Real_t> hostState;
    std::vector<std::uint32_t> hostTypes;
    simulation.export_particle_state(hostState, hostTypes);

    impl_->current = 0;
    bool ok = cuda_ok(cudaMemcpyAsync(impl_->state[0], hostState.data(),
                                      hostState.size() * sizeof(Real_t),
                                      cudaMemcpyHostToDevice, impl_->stream),
                       "cudaMemcpyAsync(state H2D)") &&
              cuda_ok(cudaMemcpyAsync(impl_->type[0], hostTypes.data(),
                                      hostTypes.size() * sizeof(std::uint32_t),
                                      cudaMemcpyHostToDevice, impl_->stream),
                       "cudaMemcpyAsync(type H2D)") &&
              cuda_ok(cudaStreamSynchronize(impl_->stream),
                       "cudaStreamSynchronize(upload)");

    if (!ok || !impl_->initialize_acceleration(simulation.config()) ||
        !impl_->sync_render_buffers()) {
        impl_->failed = true;
    }
}

void CudaComputeBackend::download_state(Simulation& simulation) {
    if (!valid())
        return;

    std::vector<Real_t> hostState(impl_->particleCount * 4);
    std::vector<std::uint32_t> hostTypes(impl_->particleCount);

    bool ok = cuda_ok(cudaMemcpyAsync(hostState.data(), impl_->state[impl_->current],
                                      hostState.size() * sizeof(Real_t),
                                      cudaMemcpyDeviceToHost, impl_->stream),
                       "cudaMemcpyAsync(state D2H)") &&
              cuda_ok(cudaMemcpyAsync(hostTypes.data(), impl_->type[impl_->current],
                                      hostTypes.size() * sizeof(std::uint32_t),
                                      cudaMemcpyDeviceToHost, impl_->stream),
                       "cudaMemcpyAsync(type D2H)") &&
              cuda_ok(cudaStreamSynchronize(impl_->stream),
                       "cudaStreamSynchronize(download)");
    if (!ok) {
        impl_->failed = true;
        return;
    }

    simulation.import_particle_state(hostState, &hostTypes);
}

void CudaComputeBackend::step(const SimulationConfig& config, Real_t dt) {
    step_many(config, dt, 1);
}

void CudaComputeBackend::step_many(const SimulationConfig& config,
                                   Real_t dt, int steps) {
    if (!valid() || steps <= 0 || impl_->particleCount == 0)
        return;

    impl_->poll_timers();

    const DeviceParams p = impl_->make_params(config, dt);
    const float halfDt = 0.5f * dt;
    const bool periodic = config.world.boundary == BoundaryMode::Periodic;

    bool timing = false;
    const std::size_t timerSlot = impl_->timerWrite;
    if (!impl_->timerPending[timerSlot]) {
        cudaEventRecord(impl_->timerStart[timerSlot], impl_->stream);
        timing = true;
    }

    bool ok = impl_->clear_counts();
    if (!ok) {
        impl_->failed = true;
        return;
    }

    // First step: current state performs the first half-kick, drift and cell count.
    if (periodic) {
        kick_drift_count_kernel<true>
            <<<impl_->particle_blocks(), impl_->particleThreads, 0, impl_->stream>>>(
                impl_->state[impl_->current], impl_->acceleration,
                impl_->cellCount, p, halfDt, dt);
    } else {
        kick_drift_count_kernel<false>
            <<<impl_->particle_blocks(), impl_->particleThreads, 0, impl_->stream>>>(
                impl_->state[impl_->current], impl_->acceleration,
                impl_->cellCount, p, halfDt, dt);
    }

    for (int stepIndex = 0; stepIndex < steps; ++stepIndex) {
        if (!impl_->scan_counts() || !impl_->clear_write_counters()) {
            impl_->failed = true;
            return;
        }

        const int next = 1 - impl_->current;
        scatter_reorder_kernel
            <<<impl_->particle_blocks(), impl_->particleThreads, 0, impl_->stream>>>(
                impl_->state[impl_->current], impl_->type[impl_->current],
                impl_->state[next], impl_->type[next],
                impl_->cellCount, impl_->cellOffset, p);

        if (periodic)
            impl_->launch_force<true>(impl_->state[next], impl_->type[next], p);
        else
            impl_->launch_force<false>(impl_->state[next], impl_->type[next], p);

        if (stepIndex + 1 < steps) {
            if (!impl_->clear_counts()) {
                impl_->failed = true;
                return;
            }

            if (periodic) {
                finish_begin_next_count_kernel<true>
                    <<<impl_->particle_blocks(), impl_->particleThreads, 0,
                       impl_->stream>>>(
                        impl_->state[next], impl_->type[next],
                        impl_->acceleration, impl_->relaxation,
                        impl_->cellCount, p, halfDt, dt);
            } else {
                finish_begin_next_count_kernel<false>
                    <<<impl_->particle_blocks(), impl_->particleThreads, 0,
                       impl_->stream>>>(
                        impl_->state[next], impl_->type[next],
                        impl_->acceleration, impl_->relaxation,
                        impl_->cellCount, p, halfDt, dt);
            }
            impl_->current = next;
        } else {
            kick_damping_kernel
                <<<impl_->particle_blocks(), impl_->particleThreads, 0,
                   impl_->stream>>>(
                    impl_->state[next], impl_->type[next],
                    impl_->acceleration, impl_->relaxation,
                    p, halfDt);
            impl_->current = next;
        }
    }

    if (timing) {
        cudaEventRecord(impl_->timerStop[timerSlot], impl_->stream);
        impl_->timerPending[timerSlot] = true;
        impl_->timerSteps[timerSlot] = steps;
        impl_->timerWrite = (impl_->timerWrite + 1) % TIMER_SLOTS;
    }

    if (!cuda_ok(cudaGetLastError(), "CUDA optimized simulation kernels") ||
        !impl_->sync_render_buffers()) {
        impl_->failed = true;
    }
}

#endif // PARTICLELIFE_ENABLE_CUDA
