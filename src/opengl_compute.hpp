#pragma once

#include <glad/glad.h>

#include "embedded_resources.hpp"
#include "simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class OpenGLComputeBackend {
  public:
    static constexpr GLuint             DEFAULT_LOCAL_SIZE = 256;
    static constexpr std::array<int, 4> LOCAL_SIZE_OPTIONS = {64, 128, 256, 512};

    bool initialize(const Simulation& simulation,
                    GLuint            localSize = DEFAULT_LOCAL_SIZE) {
        if (program_)
            return true;

        localSize_ = localSize;
        program_   = load_compute_program("resources/shaders/simulation.comp", localSize_);
        if (!program_)
            return false;

        cache_uniform_locations();

        glGenBuffers(1, &particleBuffer_);
        glGenBuffers(1, &scratchParticleBuffer_);
        glGenBuffers(1, &typeBuffer_);
        glGenBuffers(1, &scratchTypeBuffer_);
        glGenBuffers(1, &accelerationBuffer_);
        glGenBuffers(1, &localRelaxationBuffer_);
        glGenBuffers(1, &cellCursorBuffer_);
        glGenBuffers(1, &cellOffsetBuffer_);
        glGenQueries(TIMER_QUERY_COUNT, timerQueries_.data());

        rebuild(simulation);
        return true;
    }

    void destroy() {
        const GLuint buffers[] = {
            particleBuffer_, scratchParticleBuffer_, typeBuffer_, scratchTypeBuffer_,
            accelerationBuffer_, localRelaxationBuffer_,
            cellCursorBuffer_, cellOffsetBuffer_};
        glDeleteBuffers(static_cast<GLsizei>(sizeof(buffers) / sizeof(buffers[0])),
                        buffers);

        if (program_)
            glDeleteProgram(program_);
        glDeleteQueries(TIMER_QUERY_COUNT, timerQueries_.data());

        timerQueries_.fill(0);
        timerPending_.fill(false);
        particleBuffer_ = scratchParticleBuffer_ = 0;
        typeBuffer_ = scratchTypeBuffer_ = 0;
        accelerationBuffer_ = localRelaxationBuffer_ = 0;
        cellCursorBuffer_ = cellOffsetBuffer_ = 0;
        program_                              = 0;
        particleCount_                        = 0;
        cellCount_                            = 0;
    }

    bool   valid() const noexcept { return program_ != 0; }
    GLuint particle_buffer() const noexcept { return particleBuffer_; }
    GLuint type_buffer() const noexcept { return typeBuffer_; }
    GLuint local_size() const noexcept { return localSize_; }
    double last_gpu_ms() const noexcept { return lastGpuMs_; }
    int    last_gpu_steps() const noexcept { return lastGpuSteps_; }

    bool set_local_size(GLuint localSize, const SimulationConfig& config) {
        if (localSize == localSize_)
            return true;

        GLuint newProgram = load_compute_program(
            "resources/shaders/simulation.comp", localSize);
        if (!newProgram)
            return false;

        if (program_)
            glDeleteProgram(program_);
        program_   = newProgram;
        localSize_ = localSize;
        cache_uniform_locations();
        bind_particle_buffers();
        upload_uniforms(config, config.time.dt);
        return true;
    }

    void rebuild(const Simulation& simulation) {
        const SimulationConfig& config = simulation.config();
        particleCount_                 = simulation.particles().particle_count();

        std::vector<Real_t>        state;
        std::vector<std::uint32_t> types;
        simulation.export_particle_state(state, types);

        bind_data_buffer(particleBuffer_, 0,
                         state.size() * sizeof(Real_t), state.data(), GL_DYNAMIC_COPY);
        bind_data_buffer(typeBuffer_, 1,
                         types.size() * sizeof(std::uint32_t), types.data(), GL_DYNAMIC_COPY);
        allocate_buffer(scratchParticleBuffer_, 6,
                        state.size() * sizeof(Real_t), GL_DYNAMIC_COPY);
        allocate_buffer(scratchTypeBuffer_, 7,
                        types.size() * sizeof(std::uint32_t), GL_DYNAMIC_COPY);

        allocate_buffer(accelerationBuffer_, 2,
                        particleCount_ * 2 * sizeof(Real_t), GL_DYNAMIC_COPY);
        allocate_buffer(localRelaxationBuffer_, 3,
                        particleCount_ * 4 * sizeof(Real_t), GL_DYNAMIC_COPY);

        rebuild_grid_buffers(config);
        bind_particle_buffers();
        upload_uniforms(config, config.time.dt);
        initialize_acceleration(config);
    }

    void upload_state(const Simulation& simulation) {
        std::vector<Real_t>        state;
        std::vector<std::uint32_t> types;
        simulation.export_particle_state(state, types);

        if (simulation.particles().particle_count() != particleCount_) {
            rebuild(simulation);
            return;
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer_);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        static_cast<GLsizeiptr>(state.size() * sizeof(Real_t)),
                        state.data());

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, typeBuffer_);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        static_cast<GLsizeiptr>(types.size() * sizeof(std::uint32_t)),
                        types.data());

        bind_particle_buffers();
        upload_uniforms(simulation.config(), simulation.config().time.dt);
        initialize_acceleration(simulation.config());
    }

    void download_state(Simulation& simulation) const {
        if (!particleBuffer_ || particleCount_ == 0)
            return;

        std::vector<Real_t>        state(particleCount_ * 4);
        std::vector<std::uint32_t> types(particleCount_);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleBuffer_);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(state.size() * sizeof(Real_t)),
                           state.data());

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, typeBuffer_);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(types.size() * sizeof(std::uint32_t)),
                           types.data());

        simulation.import_particle_state(state, &types);
    }

    void step(const SimulationConfig& config, Real_t dt) {
        step_many(config, dt, 1);
    }

    void step_many(const SimulationConfig& config, Real_t dt, int steps) {
        if (!program_ || particleCount_ == 0 || steps <= 0)
            return;

        upload_step_uniforms(config, dt);
        glUseProgram(program_);
        bind_particle_buffers();

        poll_timer_queries();
        bool              timing    = false;
        const std::size_t timerSlot = timerWrite_;
        if (!timerPending_[timerSlot]) {
            glBeginQuery(GL_TIME_ELAPSED, timerQueries_[timerSlot]);
            timing = true;
        }

        // First step: current state performs kick/drift and grid counting.
        clear_grid_counts();
        dispatch_particle_stage(0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        dispatch_prefix_stage();
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Scatter physically reorders state/type into the scratch buffers.
        dispatch_particle_stage(3);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        dispatch_particle_stage(4);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Later steps fuse the previous finish with the next begin. Stage 6
        // runs on the spatially reordered scratch state, counts its new cells,
        // then that buffer becomes the new current state without a copy.
        for (int stepIndex = 1; stepIndex < steps; ++stepIndex) {
            clear_grid_counts();
            dispatch_particle_stage(6);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            swap_particle_buffers();

            dispatch_prefix_stage();
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            dispatch_particle_stage(3);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            dispatch_particle_stage(4);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        // Finish the final step in the reordered scratch state, then make it
        // current. No particle copy is performed; rendering reads this buffer.
        dispatch_particle_stage(5);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        swap_particle_buffers();

        if (timing) {
            glEndQuery(GL_TIME_ELAPSED);
            timerPending_[timerSlot] = true;
            timerSteps_[timerSlot]   = steps;
            timerWrite_              = (timerWrite_ + 1u) % TIMER_QUERY_COUNT;
        }
    }

  private:
    static std::string load_text(const char* path) {
#ifdef PARTICLELIFE_EMBED_RESOURCES

        std::string source = EmbeddedResources::load_text(path);

        if (source.empty())
            std::cerr << "Embedded compute shader not found: "
                      << path << '\n';

        return source;

#else

        std::ifstream file(path);
        if (!file) {
            std::cerr << "Cannot open compute shader: "
                      << path << '\n';
            return {};
        }

        std::stringstream ss;
        ss << file.rdbuf();
        return ss.str();

#endif
    }

    static std::string inject_local_size(std::string source, GLuint localSize) {
        const std::size_t lineEnd = source.find('\n');
        if (lineEnd == std::string::npos)
            return source;

        const std::string define =
            "\n#define LOCAL_SIZE " + std::to_string(localSize) + "\n";
        source.insert(lineEnd + 1, define);
        return source;
    }

    static GLuint load_compute_program(const char* path, GLuint localSize) {
        std::string source = load_text(path);
        if (source.empty())
            return 0;
        source = inject_local_size(std::move(source), localSize);

        const char* src    = source.c_str();
        GLuint      shader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);

        GLint ok = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[4096];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            std::cerr << "Compute shader compile error (local size "
                      << localSize << "):\n"
                      << log << '\n';
            glDeleteShader(shader);
            return 0;
        }

        GLuint program = glCreateProgram();
        glAttachShader(program, shader);
        glLinkProgram(program);
        glDeleteShader(shader);

        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[4096];
            glGetProgramInfoLog(program, sizeof(log), nullptr, log);
            std::cerr << "Compute shader link error (local size "
                      << localSize << "):\n"
                      << log << '\n';
            glDeleteProgram(program);
            return 0;
        }
        return program;
    }

    static void bind_data_buffer(GLuint buffer, GLuint binding,
                                 std::size_t bytes, const void* data,
                                 GLenum usage) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes),
                     data, usage);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
    }

    static void allocate_buffer(GLuint buffer, GLuint binding,
                                std::size_t bytes, GLenum usage) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes),
                     nullptr, usage);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
    }

    void bind_particle_buffers() const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleBuffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, typeBuffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, scratchParticleBuffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, scratchTypeBuffer_);
    }

    void swap_particle_buffers() {
        std::swap(particleBuffer_, scratchParticleBuffer_);
        std::swap(typeBuffer_, scratchTypeBuffer_);
        bind_particle_buffers();
    }

    void rebuild_grid_buffers(const SimulationConfig& config) {
        const WorldConfig& world = config.world;
        gridCountX_              = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(world.width / world.bucketSize));
        gridCountY_ = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(world.height / world.bucketSize));
        cellCount_ = gridCountX_ * gridCountY_;

        cellWidth_      = world.width / static_cast<Real_t>(gridCountX_);
        cellHeight_     = world.height / static_cast<Real_t>(gridCountY_);
        neighborRangeX_ = static_cast<int>(
            std::ceil(INTERACTION_RADIUS / cellWidth_));
        neighborRangeY_ = static_cast<int>(
            std::ceil(INTERACTION_RADIUS / cellHeight_));

        allocate_buffer(cellCursorBuffer_, 4,
                        static_cast<std::size_t>(cellCount_) * sizeof(std::uint32_t),
                        GL_DYNAMIC_COPY);
        allocate_buffer(cellOffsetBuffer_, 5,
                        static_cast<std::size_t>(cellCount_ + 1u) *
                            sizeof(std::uint32_t),
                        GL_DYNAMIC_COPY);
    }

    void clear_grid_counts() const {
        const std::uint32_t zero = 0u;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellCursorBuffer_);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI,
                          GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    }

    void initialize_acceleration(const SimulationConfig& config) {
        if (particleCount_ == 0)
            return;

        upload_step_uniforms(config, config.time.dt);
        glUseProgram(program_);
        bind_particle_buffers();

        clear_grid_counts();
        dispatch_particle_stage(1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        dispatch_prefix_stage();
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        dispatch_particle_stage(3);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        dispatch_particle_stage(4);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

        // Force was evaluated in physical cell order, so the reordered state
        // becomes canonical and acceleration already matches its indices.
        swap_particle_buffers();
    }

    void dispatch_particle_stage(int stage) const {
        glUniform1i(uStage_, stage);
        const GLuint groups = static_cast<GLuint>(
            (particleCount_ + localSize_ - 1u) / localSize_);
        glDispatchCompute(groups, 1, 1);
    }

    void dispatch_prefix_stage() const {
        glUniform1i(uStage_, 2);
        glDispatchCompute(1, 1, 1);
    }

    void upload_uniforms(const SimulationConfig& config, Real_t dt) {
        glUseProgram(program_);

        glUniform1ui(uParticleCount_, static_cast<GLuint>(particleCount_));
        glUniform1ui(uTypeCount_, static_cast<GLuint>(config.particleTypes.size()));
        glUniform1ui(uCellCount_, cellCount_);
        glUniform2ui(uGridCount_, gridCountX_, gridCountY_);
        glUniform2f(uInvCellSize_, 1.0f / cellWidth_, 1.0f / cellHeight_);
        glUniform2i(uNeighborRange_, neighborRangeX_, neighborRangeY_);
        glUniform2f(uWorldSize_, config.world.width, config.world.height);
        glUniform2f(uHalfWorldSize_, 0.5f * config.world.width,
                    0.5f * config.world.height);
        glUniform1i(uBoundaryMode_,
                    config.world.boundary == BoundaryMode::Periodic ? 0 : 1);
        glUniform1f(uRestitution_, config.world.restitution);
        glUniform1f(uCoreStrength_, config.physics.coreStrength);

        const Real_t coreRadius2 =
            config.physics.coreRadius * config.physics.coreRadius;
        glUniform1f(uCoreRadius2_, coreRadius2);
        glUniform1f(uInteractionSpanInv_, 1.0f / (1.0f - coreRadius2));

        std::array<Real_t, 16> invMass{};
        for (Index_t i = 0; i < config.particleTypes.size() && i < invMass.size(); ++i)
            invMass[i] = 1.0f / config.particleTypes[i].mass;
        glUniform1fv(uInvMass_, static_cast<GLsizei>(invMass.size()), invMass.data());

        std::array<Real_t, 256> interaction{};
        const std::size_t       count =
            std::min(interaction.size(), config.interaction.data.size());
        std::copy_n(config.interaction.data.data(), count, interaction.data());
        glUniform1fv(uInteraction_, static_cast<GLsizei>(interaction.size()),
                     interaction.data());

        upload_step_uniforms(config, dt);
    }

    void upload_step_uniforms(const SimulationConfig& config, Real_t dt) const {
        glUseProgram(program_);
        glUniform1f(uDt_, dt);
        glUniform1f(uHalfDt_, 0.5f * dt);
        glUniform1f(uGlobalDampingFactor_,
                    std::exp(-config.physics.damping * dt));
        glUniform1f(uLocalDampingDt2_,
                    2.0f * config.physics.localDamping * dt);
    }

    void poll_timer_queries() {
        for (std::size_t i = 0; i < TIMER_QUERY_COUNT; ++i) {
            if (!timerPending_[i])
                continue;

            GLint available = GL_FALSE;
            glGetQueryObjectiv(timerQueries_[i], GL_QUERY_RESULT_AVAILABLE, &available);
            if (!available)
                continue;

            GLuint64 elapsedNs = 0;
            glGetQueryObjectui64v(timerQueries_[i], GL_QUERY_RESULT, &elapsedNs);
            lastGpuMs_       = static_cast<double>(elapsedNs) * 1.0e-6;
            lastGpuSteps_    = timerSteps_[i];
            timerPending_[i] = false;
        }
    }

    void cache_uniform_locations() {
        uStage_         = glGetUniformLocation(program_, "uStage");
        uParticleCount_ = glGetUniformLocation(program_, "uParticleCount");
        uTypeCount_     = glGetUniformLocation(program_, "uTypeCount");
        uCellCount_     = glGetUniformLocation(program_, "uCellCount");
        uGridCount_     = glGetUniformLocation(program_, "uGridCount");
        uInvCellSize_   = glGetUniformLocation(program_, "uInvCellSize");
        uNeighborRange_ = glGetUniformLocation(program_, "uNeighborRange");
        uWorldSize_     = glGetUniformLocation(program_, "uWorldSize");
        uHalfWorldSize_ = glGetUniformLocation(program_, "uHalfWorldSize");
        uBoundaryMode_  = glGetUniformLocation(program_, "uBoundaryMode");
        uRestitution_   = glGetUniformLocation(program_, "uRestitution");
        uDt_            = glGetUniformLocation(program_, "uDt");
        uHalfDt_        = glGetUniformLocation(program_, "uHalfDt");
        uGlobalDampingFactor_ =
            glGetUniformLocation(program_, "uGlobalDampingFactor");
        uLocalDampingDt2_ = glGetUniformLocation(program_, "uLocalDampingDt2");
        uCoreRadius2_     = glGetUniformLocation(program_, "uCoreRadius2");
        uInteractionSpanInv_ =
            glGetUniformLocation(program_, "uInteractionSpanInv");
        uCoreStrength_ = glGetUniformLocation(program_, "uCoreStrength");
        uInvMass_      = glGetUniformLocation(program_, "uInvMass[0]");
        uInteraction_  = glGetUniformLocation(program_, "uInteraction[0]");
    }

  private:
    GLuint program_               = 0;
    GLuint particleBuffer_        = 0;
    GLuint scratchParticleBuffer_ = 0;
    GLuint typeBuffer_            = 0;
    GLuint scratchTypeBuffer_     = 0;
    GLuint accelerationBuffer_    = 0;
    GLuint localRelaxationBuffer_ = 0;
    GLuint cellCursorBuffer_      = 0;
    GLuint cellOffsetBuffer_      = 0;

    GLuint        localSize_      = DEFAULT_LOCAL_SIZE;
    std::size_t   particleCount_  = 0;
    std::uint32_t gridCountX_     = 1;
    std::uint32_t gridCountY_     = 1;
    std::uint32_t cellCount_      = 1;
    Real_t        cellWidth_      = 1.0f;
    Real_t        cellHeight_     = 1.0f;
    int           neighborRangeX_ = 1;
    int           neighborRangeY_ = 1;

    static constexpr std::size_t          TIMER_QUERY_COUNT = 4;
    std::array<GLuint, TIMER_QUERY_COUNT> timerQueries_{};
    std::array<bool, TIMER_QUERY_COUNT>   timerPending_{};
    std::array<int, TIMER_QUERY_COUNT>    timerSteps_{};
    std::size_t                           timerWrite_   = 0;
    double                                lastGpuMs_    = -1.0;
    int                                   lastGpuSteps_ = 0;

    GLint uStage_               = -1;
    GLint uParticleCount_       = -1;
    GLint uTypeCount_           = -1;
    GLint uCellCount_           = -1;
    GLint uGridCount_           = -1;
    GLint uInvCellSize_         = -1;
    GLint uNeighborRange_       = -1;
    GLint uWorldSize_           = -1;
    GLint uHalfWorldSize_       = -1;
    GLint uBoundaryMode_        = -1;
    GLint uRestitution_         = -1;
    GLint uDt_                  = -1;
    GLint uHalfDt_              = -1;
    GLint uGlobalDampingFactor_ = -1;
    GLint uLocalDampingDt2_     = -1;
    GLint uCoreRadius2_         = -1;
    GLint uInteractionSpanInv_  = -1;
    GLint uCoreStrength_        = -1;
    GLint uInvMass_             = -1;
    GLint uInteraction_         = -1;
};
