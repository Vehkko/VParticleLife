#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using Real_t  = float;
using Index_t = std::size_t;

constexpr Real_t CORE_RADIUS        = 0.1;
constexpr Real_t INTERACTION_RADIUS = 1.0;

constexpr Real_t CORE_RADIUS_SQUARE        = CORE_RADIUS * CORE_RADIUS;
constexpr Real_t INTERACTION_RADIUS_SQUARE = INTERACTION_RADIUS * INTERACTION_RADIUS;

constexpr Real_t WORLD_HALF_WIDTH  = 10.0;
constexpr Real_t WORLD_HALF_HEIGHT = 10.0;

constexpr Real_t WORLD_WIDTH  = 2.0 * WORLD_HALF_WIDTH;
constexpr Real_t WORLD_HEIGHT = 2.0 * WORLD_HALF_HEIGHT;

constexpr Index_t TYPE_COUNT              = 7;
constexpr Index_t PARTICLE_COUNT_PER_TYPE = 1000;
constexpr Real_t  DT                      = 0.001f;
constexpr Real_t  DAMPING                 = 0.8;

class RandomReal {
  public:
    RandomReal(Real_t inf, Real_t sup) : rd_(), gen_(rd_()), distb_(inf, sup) {}

    Real_t get() {
        return distb_(gen_);
    }

    void fill_vec(std::vector<Real_t>& vec) {
        for (Index_t i = 0; i < vec.size(); ++i) {
            vec[i] = distb_(gen_);
        }
    }

  private:
    std::random_device                     rd_;
    std::mt19937                           gen_;
    std::uniform_real_distribution<Real_t> distb_;
};

struct SquareMatrix {
    SquareMatrix(Index_t n) : width(n), data(std::vector<Real_t>(n * n)) {}

    void fill_random(Real_t inf, Real_t sup) {
        RandomReal rr(inf, sup);
        rr.fill_vec(data);
    }

    Real_t& operator()(Index_t i, Index_t j) { return data[i * width + j]; }

    const Real_t& operator()(Index_t i, Index_t j) const { return data[i * width + j]; }

    Index_t             width;
    std::vector<Real_t> data;
};

class ParticleSet {
  public:
    ParticleSet(const std::vector<Index_t>& particleCountPerType)
        : typeCount_(particleCountPerType.size()),
          particleCount_(std::accumulate(particleCountPerType.begin(), particleCountPerType.end(), static_cast<Index_t>(0))),
          particleCountPerType_(particleCountPerType),

          type_(particleCount_),
          mass_(typeCount_),
          x_(particleCount_),
          y_(particleCount_),
          vx_(particleCount_),
          vy_(particleCount_) {
        auto    it      = type_.begin();
        Index_t typeIdx = 0;
        for (Index_t n : particleCountPerType_) {
            for (Index_t i = 0; i < n; ++i) {
                *it = typeIdx;
                it++;
            }
            typeIdx++;
        }
    }

    ParticleSet(Index_t typeCount, Index_t particleCountPerType) : ParticleSet(std::vector<Index_t>(typeCount, particleCountPerType)) {}

    Index_t particle_count() const noexcept { return particleCount_; }

    const Real_t* x_data() const noexcept { return x_.data(); }

    const Real_t* y_data() const noexcept { return y_.data(); }

    const Index_t type_count() const noexcept { return typeCount_; }

    const Index_t particle_count_per_type(Index_t type) const noexcept { return particleCountPerType_[type]; }

    void fill_random_mass(Real_t inf, Real_t sup) {
        RandomReal rr(inf, sup);
        rr.fill_vec(mass_);
    }

    void fill_random_pos(Real_t infX, Real_t supX, Real_t infY, Real_t supY) {
        RandomReal rx(infX, supX), ry(infY, supY);
        rx.fill_vec(x_);
        ry.fill_vec(y_);
    }

    void fill_random_vel(Real_t infVx, Real_t supVx, Real_t infVy, Real_t supVy) {
        RandomReal rx(infVx, supVx), ry(infVy, supVy);
        rx.fill_vec(vx_);
        ry.fill_vec(vy_);
    }

  public:
    friend class Pusher;

  private:
    Index_t              typeCount_;
    Index_t              particleCount_;
    std::vector<Index_t> particleCountPerType_;

    std::vector<Index_t> type_;
    std::vector<Real_t>  mass_;

    std::vector<Real_t> x_;
    std::vector<Real_t> y_;
    std::vector<Real_t> vx_;
    std::vector<Real_t> vy_;
};

class Interaction {
  public:
    Interaction(Index_t typeCount) : strength_(SquareMatrix(typeCount)) {}

    void build_random_interaction(Real_t inf, Real_t sup) { strength_.fill_random(inf, sup); }

    Real_t force_coef(Index_t type1, Index_t type2, Real_t distance2) const {
        if (distance2 < CORE_RADIUS_SQUARE)
            return (-1. / distance2 + 1. / CORE_RADIUS_SQUARE) * INTERACTION_RADIUS;

        if (distance2 > INTERACTION_RADIUS_SQUARE)
            return 0.;

        const Real_t t = (distance2 - CORE_RADIUS_SQUARE) / (INTERACTION_RADIUS_SQUARE - CORE_RADIUS_SQUARE);
        const Real_t g = 4.0 * t * (1 - t);
        return -strength_(type1, type2) * g / INTERACTION_RADIUS;
    }

  private:
    SquareMatrix strength_;
};

class Pusher {
  public:
    Pusher(ParticleSet& ps, const Interaction& it) : particleSet_(ps), interaction_(it), ax_(std::vector<Real_t>(ps.particleCount_)), ay_(std::vector<Real_t>(ps.particleCount_)) {}

    void initialize() { compute_a(); }

    void step(Real_t dt) {
        ParticleSet& ps     = particleSet_;
        const Real_t halfDt = 0.5f * dt;

        for (Index_t i = 0; i < ps.particleCount_; ++i) {
            ps.vx_[i] += ax_[i] * halfDt;
            ps.vy_[i] += ay_[i] * halfDt;

            ps.x_[i] += ps.vx_[i] * dt;
            ps.y_[i] += ps.vy_[i] * dt;

            if (ps.x_[i] >= WORLD_HALF_WIDTH)
                ps.x_[i] -= WORLD_WIDTH;
            else if (ps.x_[i] < -WORLD_HALF_WIDTH)
                ps.x_[i] += WORLD_WIDTH;

            if (ps.y_[i] >= WORLD_HALF_HEIGHT)
                ps.y_[i] -= WORLD_HEIGHT;
            else if (ps.y_[i] < -WORLD_HALF_HEIGHT)
                ps.y_[i] += WORLD_HEIGHT;
        }

        compute_a();

        const Real_t damping = std::exp(-DAMPING * dt);

        for (Index_t i = 0; i < ps.particleCount_; ++i) {
            ps.vx_[i] += ax_[i] * halfDt;
            ps.vy_[i] += ay_[i] * halfDt;

            ps.vx_[i] *= damping;
            ps.vy_[i] *= damping;
        }
    }

  private:
    void compute_a() {
        const ParticleSet& ps = particleSet_;

#pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < ps.particleCount_; i++) {
            const Index_t typeI = ps.type_[i];
            const Real_t  invM  = 1.0 / ps.mass_[typeI];
            const Real_t  xi    = ps.x_[i];
            const Real_t  yi    = ps.y_[i];

            Real_t ax = 0.0f;
            Real_t ay = 0.0f;
            for (Index_t j = 0; j < ps.particleCount_; j++) {
                if (i == j)
                    continue;

                const Index_t typeJ = ps.type_[j];
                Real_t        dx    = ps.x_[j] - xi;
                Real_t        dy    = ps.y_[j] - yi;

                if (dx > WORLD_HALF_WIDTH)
                    dx -= WORLD_WIDTH;
                else if (dx < -WORLD_HALF_WIDTH)
                    dx += WORLD_WIDTH;

                if (dy > WORLD_HALF_HEIGHT)
                    dy -= WORLD_HEIGHT;
                else if (dy < -WORLD_HALF_HEIGHT)
                    dy += WORLD_HEIGHT;

                const Real_t d2 = dx * dx + dy * dy;

                const Real_t force  = interaction_.force_coef(typeI, typeJ, d2);
                const Real_t forceX = force * dx;
                const Real_t forceY = force * dy;

                ax += forceX;
                ay += forceY;
            }

            ax_[i] = ax * invM;
            ay_[i] = ay * invM;
        }
    }

  private:
    ParticleSet&       particleSet_;
    const Interaction& interaction_;

    std::vector<Real_t> ax_;
    std::vector<Real_t> ay_;
};

static std::string load_text_file(const char* path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Cannot open file: " << path << '\n';
        return {};
    }

    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static GLuint compile_shader(GLenum type, const std::string& source) {
    GLuint      shader = glCreateShader(type);
    const char* src    = source.c_str();

    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error:\n"
                  << log << '\n';
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

static GLuint load_shader_program(const char* vertexPath, const char* fragmentPath) {
    const std::string vertexSource   = load_text_file(vertexPath);
    const std::string fragmentSource = load_text_file(fragmentPath);
    if (vertexSource.empty() || fragmentSource.empty())
        return 0;

    GLuint vertexShader   = compile_shader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compile_shader(GL_FRAGMENT_SHADER, fragmentSource);
    if (!vertexShader || !fragmentShader) {
        if (vertexShader)
            glDeleteShader(vertexShader);
        if (fragmentShader)
            glDeleteShader(fragmentShader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Shader link error:\n"
                  << log << '\n';
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

struct Color {
    Real_t r, g, b, a;
};

constexpr Color COLORS[] = {
    {1.00f, 0.25f, 0.25f, 0.70f},
    {0.25f, 0.65f, 1.00f, 0.70f},
    {0.25f, 1.00f, 0.40f, 0.70f},
    {1.00f, 0.75f, 0.20f, 0.70f},
    {0.80f, 0.35f, 1.00f, 0.70f},
    {0.20f, 1.00f, 0.90f, 0.70f},
    {1.00f, 0.35f, 0.70f, 0.70f},
    {0.75f, 0.80f, 1.00f, 0.70f},
};

constexpr Index_t COLOR_COUNT = sizeof(COLORS) / sizeof(COLORS[0]);

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1200, 900, "VParticleLife", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << '\n';
    glfwSwapInterval(1);

    ParticleSet particles(TYPE_COUNT, PARTICLE_COUNT_PER_TYPE);
    particles.fill_random_mass(0.5f, 1.5f);
    particles.fill_random_pos(-WORLD_HALF_WIDTH, WORLD_HALF_WIDTH, -WORLD_HALF_HEIGHT, WORLD_HALF_HEIGHT);
    particles.fill_random_vel(-0.2f, 0.2f, -0.2f, 0.2f);

    Interaction interaction(TYPE_COUNT);
    interaction.build_random_interaction(-4.f, 4.f);

    Pusher pusher(particles, interaction);
    pusher.initialize();

    GLuint program = load_shader_program("resources/shaders/particle.vert", "resources/shaders/particle.frag");
    if (!program) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    GLint halfSizeLocation = glGetUniformLocation(program, "uHalfSize");
    GLint colorLocation    = glGetUniformLocation(program, "uColor");

    GLuint vao, vboX, vboY;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vboX);
    glGenBuffers(1, &vboY);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vboX);
    glBufferData(GL_ARRAY_BUFFER, particles.particle_count() * sizeof(Real_t), particles.x_data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vboY);
    glBufferData(GL_ARRAY_BUFFER, particles.particle_count() * sizeof(Real_t), particles.y_data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        pusher.step(DT);

        glBindBuffer(GL_ARRAY_BUFFER, vboX);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particles.particle_count() * sizeof(Real_t), particles.x_data());

        glBindBuffer(GL_ARRAY_BUFFER, vboY);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particles.particle_count() * sizeof(Real_t), particles.y_data());

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0)
            continue;

        glViewport(0, 0, width, height);

        const Real_t aspect         = static_cast<Real_t>(width) / static_cast<Real_t>(height);
        const Real_t worldHalfWidth = WORLD_HALF_HEIGHT * aspect;

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glUniform2f(halfSizeLocation, worldHalfWidth, WORLD_HALF_HEIGHT);
        glBindVertexArray(vao);

        Index_t first = 0;
        for (Index_t type = 0; type < particles.type_count(); ++type) {
            const Color&  color = COLORS[type % COLOR_COUNT];
            const Index_t count = particles.particle_count_per_type(type);

            glUniform4f(colorLocation, color.r, color.g, color.b, color.a);
            glDrawArrays(GL_POINTS, static_cast<GLint>(first), static_cast<GLsizei>(count));

            first += count;
        }

        glfwSwapBuffers(window);
    }

    glDeleteBuffers(1, &vboX);
    glDeleteBuffers(1, &vboY);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
