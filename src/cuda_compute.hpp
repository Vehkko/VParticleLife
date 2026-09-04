#pragma once

#include <glad/glad.h>

#include "simulation.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct CudaDeviceInfo {
    int id = -1;
    std::string name;
    std::uint64_t totalMemory = 0;
    int major = 0;
    int minor = 0;
    int multiprocessorCount = 0;
    int warpSize = 0;
    int maxThreadsPerBlock = 0;
    bool openGLCompatible = false;
};

#ifdef PARTICLELIFE_ENABLE_CUDA

class CudaComputeBackend {
  public:
    CudaComputeBackend();
    ~CudaComputeBackend();

    CudaComputeBackend(const CudaComputeBackend&) = delete;
    CudaComputeBackend& operator=(const CudaComputeBackend&) = delete;

    static bool compiled() noexcept { return true; }
    static std::vector<CudaDeviceInfo> enumerate_devices();

    bool initialize(const Simulation& simulation, int requestedDevice = -1);
    void destroy();

    bool valid() const noexcept;
    int device_id() const noexcept;
    const char* device_name() const noexcept;

    GLuint particle_buffer() const noexcept;
    GLuint type_buffer() const noexcept;

    double last_gpu_ms() const noexcept;
    int last_gpu_steps() const noexcept;

    // Resolved launch sizes chosen from the actual CUDA kernel resource usage.
    int particle_threads() const noexcept;
    int force_threads() const noexcept;

    void rebuild(const Simulation& simulation);
    void upload_state(const Simulation& simulation);
    void download_state(Simulation& simulation);

    void step(const SimulationConfig& config, Real_t dt);
    void step_many(const SimulationConfig& config, Real_t dt, int steps);

  private:
    struct Impl;
    Impl* impl_ = nullptr;
};

#else

class CudaComputeBackend {
  public:
    static bool compiled() noexcept { return false; }
    static std::vector<CudaDeviceInfo> enumerate_devices() { return {}; }

    bool initialize(const Simulation&, int = -1) { return false; }
    void destroy() {}
    bool valid() const noexcept { return false; }
    int device_id() const noexcept { return -1; }
    const char* device_name() const noexcept { return "CUDA backend not compiled"; }
    GLuint particle_buffer() const noexcept { return 0; }
    GLuint type_buffer() const noexcept { return 0; }
    double last_gpu_ms() const noexcept { return -1.0; }
    int last_gpu_steps() const noexcept { return 0; }
    int particle_threads() const noexcept { return 0; }
    int force_threads() const noexcept { return 0; }
    void rebuild(const Simulation&) {}
    void upload_state(const Simulation&) {}
    void download_state(Simulation&) {}
    void step(const SimulationConfig&, Real_t) {}
    void step_many(const SimulationConfig&, Real_t, int) {}
};

#endif
