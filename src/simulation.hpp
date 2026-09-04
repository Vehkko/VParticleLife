#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using Real_t          = float;
using Index_t         = std::size_t;
using ParticleIndex_t = std::uint32_t;
using ParticleType_t  = std::uint16_t;

constexpr Real_t INTERACTION_RADIUS = 1.0f;
constexpr Real_t INTERACTION_RADIUS_SQUARE =
    INTERACTION_RADIUS * INTERACTION_RADIUS;
constexpr Real_t MIN_DISTANCE_SQUARE = 1.0e-8f;

struct Color {
    Real_t r, g, b, a;
};

enum class BoundaryMode {
    Periodic,
    Reflective,
};

struct RandomConfig {
    std::uint32_t seed = 0;

    Real_t posMinX = -37.5f;
    Real_t posMaxX = 37.5f;
    Real_t posMinY = -25.0f;
    Real_t posMaxY = 25.0f;

    Real_t velMinX = -0.2f;
    Real_t velMaxX = 0.2f;
    Real_t velMinY = -0.2f;
    Real_t velMaxY = 0.2f;

    Real_t massMin = 0.5f;
    Real_t massMax = 1.5f;

    Real_t interactionMin = -4.0f;
    Real_t interactionMax = 4.0f;
};

struct TimeConfig {
    Real_t dt     = 0.001f;
    Real_t speed  = 1.0f;
    bool   paused = false;
};

struct WorldConfig {
    Real_t width      = 75.0f;
    Real_t height     = 50.0f;
    Real_t bucketSize = 1.0f;

    BoundaryMode boundary    = BoundaryMode::Periodic;
    Real_t       restitution = 1.0f;
};

struct PhysicsConfig {
    Real_t damping      = 5.0f;
    Real_t localDamping = 5.0f;
    Real_t coreRadius   = 0.2f;
    Real_t coreStrength = 5.0f;
};

struct ParticleTypeConfig {
    Index_t count = 10000;
    Real_t  mass  = 1.0f;
    Real_t  size  = 4.0f;
    Color   color = {1.0f, 1.0f, 1.0f, 0.70f};
};

enum class ComputeBackend {
    CpuOpenMP,
    OpenGLCompute,
    Cuda,
};

struct ComputeConfig {
    ComputeBackend backend    = ComputeBackend::OpenGLCompute;
    int            cpuThreads = 0; // 0 = runtime default / all available threads

    // 0 = automatic. Manual values are selected from 64/128/256/512.
    // 256 is the automatic target when the device supports it; using the
    // maximum possible work-group size is often slower because it can reduce
    // occupancy.
    int gpuWorkGroupSize = 0;

    // -1 = first CUDA device compatible with the current OpenGL context.
    int cudaDevice = -1;
};

struct DisplayConfig {
    bool vsync    = true;
    bool showGrid = false;
    int  fpsLimit = 0;

    Real_t gridOpacity   = 0.33f;
    Real_t gridThickness = 1.25f;
    Color  gridColor     = {0.59f, 0.59f, 0.59f, 1.0f};

    // Screen-space fluorescent bloom. The bloom path is rendered at half
    // resolution, so its cost depends mostly on viewport pixels rather than
    // particle count. Disabling it skips all extra passes.
    bool   glowEnabled  = true;
    Real_t glowStrength = 0.85f;
    Real_t glowRadius   = 1.75f;
    Real_t glowDensity  = 0.90f;
    Real_t glowExposure = 1.15f;
};

struct SquareMatrix {
    SquareMatrix() = default;
    explicit SquareMatrix(Index_t n) { resize(n); }

    void resize(Index_t n) {
        if (n == width)
            return;

        std::vector<Real_t> newData(n * n, 0.0f);
        const Index_t       copyWidth = std::min(width, n);

        for (Index_t i = 0; i < copyWidth; ++i)
            for (Index_t j = 0; j < copyWidth; ++j)
                newData[i * n + j] = data[i * width + j];

        width = n;
        data.swap(newData);
    }

    Real_t& operator()(Index_t i, Index_t j) { return data[i * width + j]; }

    const Real_t& operator()(Index_t i, Index_t j) const {
        return data[i * width + j];
    }

    Index_t             width = 0;
    std::vector<Real_t> data;
};

struct SimulationConfig {
    RandomConfig                    random;
    TimeConfig                      time;
    WorldConfig                     world;
    PhysicsConfig                   physics;
    DisplayConfig                   display;
    ComputeConfig                   compute;
    std::vector<ParticleTypeConfig> particleTypes;
    SquareMatrix                    interaction;
};

inline SimulationConfig make_default_config() {
    constexpr Color colors[] = {
        {1.00f, 0.25f, 0.25f, 0.70f},
        {0.25f, 0.65f, 1.00f, 0.70f},
        {0.25f, 1.00f, 0.40f, 0.70f},
        {1.00f, 0.75f, 0.20f, 0.70f},
        {0.80f, 0.35f, 1.00f, 0.70f},
        {0.20f, 1.00f, 0.90f, 0.70f},
        {1.00f, 0.35f, 0.70f, 0.70f},
        {0.75f, 0.80f, 1.00f, 0.70f},
    };

    constexpr Index_t typeCount = 8;

    SimulationConfig config;
    config.random.seed = std::random_device{}();
    config.particleTypes.resize(typeCount);
    config.interaction.resize(typeCount);

    for (Index_t i = 0; i < typeCount; ++i)
        config.particleTypes[i].color = colors[i % (sizeof(colors) / sizeof(colors[0]))];

    return config;
}

class ParticleSet {
  public:
    void reset(const std::vector<ParticleTypeConfig>& types) {
        particleCountPerType_.resize(types.size());
        for (Index_t i = 0; i < types.size(); ++i)
            particleCountPerType_[i] = types[i].count;

        particleCount_ = std::accumulate(particleCountPerType_.begin(),
                                         particleCountPerType_.end(),
                                         static_cast<Index_t>(0));

        type_.resize(particleCount_);
        x_.resize(particleCount_);
        y_.resize(particleCount_);
        vx_.resize(particleCount_);
        vy_.resize(particleCount_);

        Index_t particle = 0;
        for (Index_t type = 0; type < particleCountPerType_.size(); ++type)
            for (Index_t i = 0; i < particleCountPerType_[type]; ++i)
                type_[particle++] = static_cast<ParticleType_t>(type);
    }

    Index_t particle_count() const noexcept { return particleCount_; }
    Index_t type_count() const noexcept { return particleCountPerType_.size(); }

    Index_t particle_count_per_type(Index_t type) const noexcept {
        return particleCountPerType_[type];
    }

    const Real_t*         x_data() const noexcept { return x_.data(); }
    const Real_t*         y_data() const noexcept { return y_.data(); }
    const ParticleType_t* type_data() const noexcept { return type_.data(); }

    void export_interleaved_state(std::vector<Real_t>&        state,
                                  std::vector<std::uint32_t>& types) const {
        state.resize(particleCount_ * 4);
        types.resize(particleCount_);
        for (Index_t i = 0; i < particleCount_; ++i) {
            state[4 * i + 0] = x_[i];
            state[4 * i + 1] = y_[i];
            state[4 * i + 2] = vx_[i];
            state[4 * i + 3] = vy_[i];
            types[i]         = static_cast<std::uint32_t>(type_[i]);
        }
    }

    void import_interleaved_state(const std::vector<Real_t>&        state,
                                  const std::vector<std::uint32_t>* types = nullptr) {
        if (state.size() != particleCount_ * 4)
            return;
        if (types && types->size() != particleCount_)
            return;

        for (Index_t i = 0; i < particleCount_; ++i) {
            x_[i]  = state[4 * i + 0];
            y_[i]  = state[4 * i + 1];
            vx_[i] = state[4 * i + 2];
            vy_[i] = state[4 * i + 3];
            if (types)
                type_[i] = static_cast<ParticleType_t>((*types)[i]);
        }
    }

  private:
    friend class Pusher;
    friend class Simulation;

    Index_t                     particleCount_ = 0;
    std::vector<Index_t>        particleCountPerType_;
    std::vector<ParticleType_t> type_;
    std::vector<Real_t>         x_;
    std::vector<Real_t>         y_;
    std::vector<Real_t>         vx_;
    std::vector<Real_t>         vy_;
};

class Pusher {
  public:
    Pusher(ParticleSet& particles, const SimulationConfig& config)
        : particles_(particles), config_(config) {}

    void rebuild() {
        const WorldConfig& world = config_.world;

        gridCountX_ = std::max<Index_t>(
            1, static_cast<Index_t>(world.width / world.bucketSize));
        gridCountY_ = std::max<Index_t>(
            1, static_cast<Index_t>(world.height / world.bucketSize));

        cellWidth_  = world.width / static_cast<Real_t>(gridCountX_);
        cellHeight_ = world.height / static_cast<Real_t>(gridCountY_);

        neighborRangeX_ = static_cast<int>(
            std::ceil(INTERACTION_RADIUS / cellWidth_));
        neighborRangeY_ = static_cast<int>(
            std::ceil(INTERACTION_RADIUS / cellHeight_));

        const Index_t particleCount = particles_.particleCount_;
        const Index_t cellCount     = gridCountX_ * gridCountY_;

        ax_.assign(particleCount, 0.0f);
        ay_.assign(particleCount, 0.0f);
        localVelocityX_.assign(particleCount, 0.0f);
        localVelocityY_.assign(particleCount, 0.0f);
        localWeight_.assign(particleCount, 0.0f);

        // CPU grid uses compact 32-bit indices and contiguous per-cell ranges.
        // ParticleLife cannot practically approach 2^32 particles in memory,
        // so 64-bit linked-list indices only waste bandwidth in the hot loop.
        particleCell_.assign(particleCount, 0);
        sortedParticle_.assign(particleCount, 0);
        cellCount_.assign(cellCount, 0);
        cellOffset_.assign(cellCount + 1, 0);
        cellCursor_.assign(cellCount, 0);

        rebuild_neighbor_cells();
    }

    void initialize() { compute_a(); }

    void step(Real_t dt) {
        ParticleSet& ps     = particles_;
        const Real_t halfDt = 0.5f * dt;

#pragma omp parallel for schedule(static) if (ps.particleCount_ >= 4096)
        for (std::ptrdiff_t ii = 0;
             ii < static_cast<std::ptrdiff_t>(ps.particleCount_); ++ii) {
            const Index_t i = static_cast<Index_t>(ii);
            ps.vx_[i] += ax_[i] * halfDt;
            ps.vy_[i] += ay_[i] * halfDt;

            ps.x_[i] += ps.vx_[i] * dt;
            ps.y_[i] += ps.vy_[i] * dt;

            apply_boundary(ps.x_[i], ps.y_[i], ps.vx_[i], ps.vy_[i]);
        }

        compute_a();

        const Real_t damping      = std::exp(-config_.physics.damping * dt);
        const Real_t localDamping = config_.physics.localDamping;

#pragma omp parallel for schedule(static) if (ps.particleCount_ >= 4096)
        for (std::ptrdiff_t ii = 0;
             ii < static_cast<std::ptrdiff_t>(ps.particleCount_); ++ii) {
            const Index_t i = static_cast<Index_t>(ii);
            ps.vx_[i] += ax_[i] * halfDt;
            ps.vy_[i] += ay_[i] * halfDt;

            const Real_t localWeight = localWeight_[i];
            if (localWeight > 0.0f && localDamping > 0.0f) {
                const Index_t typeI  = static_cast<Index_t>(ps.type_[i]);
                const Real_t  mass   = config_.particleTypes[typeI].mass;
                const Real_t  meanVx = localVelocityX_[i] / localWeight;
                const Real_t  meanVy = localVelocityY_[i] / localWeight;
                const Real_t  rate   = localDamping * localWeight / mass;
                const Real_t  blend =
                    0.5f * (1.0f - std::exp(-2.0f * rate * dt));

                ps.vx_[i] += blend * (meanVx - ps.vx_[i]);
                ps.vy_[i] += blend * (meanVy - ps.vy_[i]);
            }

            ps.vx_[i] *= damping;
            ps.vy_[i] *= damping;
        }
    }

  private:
    Index_t cell_x(Real_t x) const {
        const Real_t halfWidth = 0.5f * config_.world.width;
        Index_t      cell      = static_cast<Index_t>((x + halfWidth) / cellWidth_);
        return std::min(cell, gridCountX_ - 1);
    }

    Index_t cell_y(Real_t y) const {
        const Real_t halfHeight = 0.5f * config_.world.height;
        Index_t      cell       = static_cast<Index_t>((y + halfHeight) / cellHeight_);
        return std::min(cell, gridCountY_ - 1);
    }

    void apply_boundary(Real_t& x, Real_t& y, Real_t& vx, Real_t& vy) const {
        const WorldConfig& world      = config_.world;
        const Real_t       halfWidth  = 0.5f * world.width;
        const Real_t       halfHeight = 0.5f * world.height;

        if (world.boundary == BoundaryMode::Periodic) {
            if (x >= halfWidth)
                x -= world.width;
            else if (x < -halfWidth)
                x += world.width;

            if (y >= halfHeight)
                y -= world.height;
            else if (y < -halfHeight)
                y += world.height;
            return;
        }

        if (x > halfWidth) {
            x  = 2.0f * halfWidth - x;
            vx = -vx * world.restitution;
        } else if (x < -halfWidth) {
            x  = -2.0f * halfWidth - x;
            vx = -vx * world.restitution;
        }

        if (y > halfHeight) {
            y  = 2.0f * halfHeight - y;
            vy = -vy * world.restitution;
        } else if (y < -halfHeight) {
            y  = -2.0f * halfHeight - y;
            vy = -vy * world.restitution;
        }
    }

    void rebuild_neighbor_cells() {
        const Index_t totalCells = gridCountX_ * gridCountY_;
        neighborCellOffset_.assign(totalCells + 1, 0);
        neighborCells_.clear();
        neighborCells_.reserve(totalCells *
                               static_cast<Index_t>((2 * neighborRangeX_ + 1) *
                                                    (2 * neighborRangeY_ + 1)));

        for (Index_t cellY = 0; cellY < gridCountY_; ++cellY) {
            for (Index_t cellX = 0; cellX < gridCountX_; ++cellX) {
                const Index_t cell = cellY * gridCountX_ + cellX;
                neighborCellOffset_[cell] =
                    static_cast<ParticleIndex_t>(neighborCells_.size());

                for (int oy = -neighborRangeY_; oy <= neighborRangeY_; ++oy) {
                    for (int ox = -neighborRangeX_; ox <= neighborRangeX_; ++ox) {
                        std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(cellX) + ox;
                        std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(cellY) + oy;

                        if (config_.world.boundary == BoundaryMode::Periodic) {
                            if (nx < 0)
                                nx += static_cast<std::ptrdiff_t>(gridCountX_);
                            else if (nx >= static_cast<std::ptrdiff_t>(gridCountX_))
                                nx -= static_cast<std::ptrdiff_t>(gridCountX_);

                            if (ny < 0)
                                ny += static_cast<std::ptrdiff_t>(gridCountY_);
                            else if (ny >= static_cast<std::ptrdiff_t>(gridCountY_))
                                ny -= static_cast<std::ptrdiff_t>(gridCountY_);
                        } else if (nx < 0 ||
                                   nx >= static_cast<std::ptrdiff_t>(gridCountX_) ||
                                   ny < 0 ||
                                   ny >= static_cast<std::ptrdiff_t>(gridCountY_)) {
                            continue;
                        }

                        neighborCells_.push_back(static_cast<ParticleIndex_t>(
                            static_cast<Index_t>(ny) * gridCountX_ +
                            static_cast<Index_t>(nx)));
                    }
                }
            }
        }

        neighborCellOffset_[totalCells] =
            static_cast<ParticleIndex_t>(neighborCells_.size());
    }

    void build_grid() {
        const ParticleSet& ps = particles_;
        std::fill(cellCount_.begin(), cellCount_.end(), 0u);

        for (Index_t i = 0; i < ps.particleCount_; ++i) {
            const Index_t cell =
                cell_y(ps.y_[i]) * gridCountX_ + cell_x(ps.x_[i]);
            particleCell_[i] = static_cast<ParticleIndex_t>(cell);
            ++cellCount_[cell];
        }

        cellOffset_[0] = 0;
        for (Index_t cell = 0; cell < cellCount_.size(); ++cell)
            cellOffset_[cell + 1] = cellOffset_[cell] + cellCount_[cell];

        std::copy(cellOffset_.begin(), cellOffset_.end() - 1,
                  cellCursor_.begin());

        for (Index_t i = 0; i < ps.particleCount_; ++i) {
            const ParticleIndex_t cell = particleCell_[i];
            sortedParticle_[cellCursor_[cell]++] =
                static_cast<ParticleIndex_t>(i);
        }
    }

    template <bool Periodic>
    void compute_a_impl() {
        const ParticleSet&   ps          = particles_;
        const WorldConfig&   world       = config_.world;
        const PhysicsConfig& physics     = config_.physics;
        const Real_t         halfWidth   = 0.5f * world.width;
        const Real_t         halfHeight  = 0.5f * world.height;
        const Real_t         coreRadius2 = physics.coreRadius * physics.coreRadius;
        const Real_t         interactionScale =
            4.0f / (INTERACTION_RADIUS_SQUARE - coreRadius2);
        const Index_t interactionWidth = config_.interaction.width;

#pragma omp parallel for schedule(guided, 64)
        for (std::ptrdiff_t ii = 0;
             ii < static_cast<std::ptrdiff_t>(ps.particleCount_); ++ii) {
            // Traverse targets in cell order. This improves cache locality and,
            // together with guided scheduling, prevents dense moving clusters
            // from pinning one OpenMP worker with a disproportionate workload.
            const Index_t i = static_cast<Index_t>(
                sortedParticle_[static_cast<Index_t>(ii)]);
            const Index_t typeI = static_cast<Index_t>(ps.type_[i]);
            const Real_t  invM  = 1.0f / config_.particleTypes[typeI].mass;
            const Real_t* interactionRow =
                config_.interaction.data.data() + typeI * interactionWidth;
            const Real_t xi = ps.x_[i];
            const Real_t yi = ps.y_[i];

            Real_t ax             = 0.0f;
            Real_t ay             = 0.0f;
            Real_t localVelocityX = 0.0f;
            Real_t localVelocityY = 0.0f;
            Real_t localWeight    = 0.0f;

            const ParticleIndex_t cell          = particleCell_[i];
            const ParticleIndex_t neighborBegin = neighborCellOffset_[cell];
            const ParticleIndex_t neighborEnd   = neighborCellOffset_[cell + 1];

            for (ParticleIndex_t ns = neighborBegin; ns < neighborEnd; ++ns) {
                const ParticleIndex_t neighborCell = neighborCells_[ns];
                const ParticleIndex_t begin        = cellOffset_[neighborCell];
                const ParticleIndex_t end          = cellOffset_[neighborCell + 1];

                for (ParticleIndex_t slot = begin; slot < end; ++slot) {
                    const Index_t j = sortedParticle_[slot];
                    if (i == j)
                        continue;

                    Real_t dx = ps.x_[j] - xi;
                    Real_t dy = ps.y_[j] - yi;

                    if constexpr (Periodic) {
                        if (dx > halfWidth)
                            dx -= world.width;
                        else if (dx < -halfWidth)
                            dx += world.width;

                        if (dy > halfHeight)
                            dy -= world.height;
                        else if (dy < -halfHeight)
                            dy += world.height;
                    }

                    const Real_t d2 = dx * dx + dy * dy;
                    if (d2 > INTERACTION_RADIUS_SQUARE)
                        continue;

                    Real_t force;
                    if (d2 < coreRadius2) {
                        const Real_t safeDistance2 =
                            std::max(d2, MIN_DISTANCE_SQUARE);
                        force = -physics.coreStrength / safeDistance2 *
                                INTERACTION_RADIUS;

                        const Real_t weight = 1.0f - d2 / coreRadius2;
                        localVelocityX += weight * ps.vx_[j];
                        localVelocityY += weight * ps.vy_[j];
                        localWeight += weight;
                    } else {
                        const Index_t typeJ = static_cast<Index_t>(ps.type_[j]);
                        const Real_t  t     = (d2 - coreRadius2) /
                                              (INTERACTION_RADIUS_SQUARE - coreRadius2);
                        const Real_t  g     = interactionScale *
                                              (d2 - coreRadius2) * (1.0f - t);
                        force               = -interactionRow[typeJ] * g / INTERACTION_RADIUS;
                    }

                    ax += force * dx;
                    ay += force * dy;
                }
            }

            ax_[i]             = ax * invM;
            ay_[i]             = ay * invM;
            localVelocityX_[i] = localVelocityX;
            localVelocityY_[i] = localVelocityY;
            localWeight_[i]    = localWeight;
        }
    }

    void compute_a() {
        build_grid();
        if (config_.world.boundary == BoundaryMode::Periodic)
            compute_a_impl<true>();
        else
            compute_a_impl<false>();
    }

  private:
    ParticleSet&            particles_;
    const SimulationConfig& config_;

    std::vector<Real_t> ax_;
    std::vector<Real_t> ay_;
    std::vector<Real_t> localVelocityX_;
    std::vector<Real_t> localVelocityY_;
    std::vector<Real_t> localWeight_;

    std::vector<ParticleIndex_t> particleCell_;
    std::vector<ParticleIndex_t> sortedParticle_;
    std::vector<ParticleIndex_t> cellCount_;
    std::vector<ParticleIndex_t> cellOffset_;
    std::vector<ParticleIndex_t> cellCursor_;
    std::vector<ParticleIndex_t> neighborCellOffset_;
    std::vector<ParticleIndex_t> neighborCells_;

    Index_t gridCountX_     = 0;
    Index_t gridCountY_     = 0;
    Real_t  cellWidth_      = 1.0f;
    Real_t  cellHeight_     = 1.0f;
    int     neighborRangeX_ = 1;
    int     neighborRangeY_ = 1;
};

class Simulation {
  public:
    explicit Simulation(SimulationConfig config)
        : config_(std::move(config)), pusher_(particles_, config_), rng_(config_.random.seed) {
        rebuild();
        randomize_all();
    }

    void rebuild() {
        config_.interaction.resize(config_.particleTypes.size());
        particles_.reset(config_.particleTypes);
        pusher_.rebuild();
    }

    void rebuild_grid(bool refreshAcceleration = true) {
        pusher_.rebuild();
        if (refreshAcceleration)
            pusher_.initialize();
    }

    void randomize_all(bool refreshAcceleration = true) {
        randomize_selected(true, true, true, true, refreshAcceleration);
    }

    void randomize_particles(bool refreshAcceleration = true) {
        randomize_selected(true, true, false, false, refreshAcceleration);
    }

    void randomize_selected(bool position, bool velocity, bool mass,
                            bool interaction, bool refreshAcceleration = true) {
        rng_.seed(config_.random.seed);

        if (position) {
            const Real_t halfWidth  = 0.5f * config_.world.width;
            const Real_t halfHeight = 0.5f * config_.world.height;

            const Real_t minX = std::clamp(config_.random.posMinX, -halfWidth, halfWidth);
            const Real_t maxX = std::clamp(config_.random.posMaxX, -halfWidth, halfWidth);
            const Real_t minY = std::clamp(config_.random.posMinY, -halfHeight, halfHeight);
            const Real_t maxY = std::clamp(config_.random.posMaxY, -halfHeight, halfHeight);

            fill_random(particles_.x_, minX, maxX);
            fill_random(particles_.y_, minY, maxY);
        }

        if (velocity) {
            fill_random(particles_.vx_, config_.random.velMinX, config_.random.velMaxX);
            fill_random(particles_.vy_, config_.random.velMinY, config_.random.velMaxY);
        }

        if (mass) {
            for (ParticleTypeConfig& type : config_.particleTypes)
                type.mass = random_real(config_.random.massMin, config_.random.massMax);
        }

        if (interaction) {
            for (Real_t& value : config_.interaction.data)
                value = random_real(config_.random.interactionMin,
                                    config_.random.interactionMax);
        }

        if (refreshAcceleration)
            pusher_.initialize();
    }

    void refresh_acceleration() { pusher_.initialize(); }

    void step() { pusher_.step(config_.time.dt); }
    void step(Real_t dt) { pusher_.step(dt); }

    SimulationConfig&       config() noexcept { return config_; }
    const SimulationConfig& config() const noexcept { return config_; }

    const ParticleSet& particles() const noexcept { return particles_; }

    void export_particle_state(std::vector<Real_t>&        state,
                               std::vector<std::uint32_t>& types) const {
        particles_.export_interleaved_state(state, types);
    }

    void import_particle_state(const std::vector<Real_t>&        state,
                               const std::vector<std::uint32_t>* types = nullptr) {
        particles_.import_interleaved_state(state, types);
    }

  private:
    Real_t random_real(Real_t inf, Real_t sup) {
        if (inf > sup)
            std::swap(inf, sup);
        std::uniform_real_distribution<Real_t> dist(inf, sup);
        return dist(rng_);
    }

    void fill_random(std::vector<Real_t>& values, Real_t inf, Real_t sup) {
        if (inf > sup)
            std::swap(inf, sup);
        std::uniform_real_distribution<Real_t> dist(inf, sup);
        for (Real_t& value : values)
            value = dist(rng_);
    }

  private:
    SimulationConfig config_;
    ParticleSet      particles_;
    Pusher           pusher_;
    std::mt19937     rng_;
};
