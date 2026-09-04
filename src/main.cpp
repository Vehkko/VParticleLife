#include <ratio>
#include <system_error>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dxgi1_2.h>
#include <windows.h>
#endif

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "cuda_compute.hpp"
#include "opengl_compute.hpp"
#include "simulation.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr float PANEL_DEFAULT_WIDTH = 500.0f;
constexpr float PANEL_MIN_WIDTH = 260.0f;
constexpr float PANEL_MAX_WIDTH = 720.0f;
constexpr float PANEL_COLLAPSE_THRESHOLD = 90.0f;
constexpr float PANEL_HANDLE_WIDTH = 12.0f;
constexpr float CAMERA_MIN_SCALE = 1.0f;
constexpr float CAMERA_MAX_SCALE = 2000.0f;
// ParticleTypeConfig::size keeps the familiar old UI scale: at 20 pixels per
// world unit, size=6 renders as a 6 px point. Camera zoom multiplies it from
// there, so particles scale together with the world instead of staying
// screen-sized.
constexpr float PARTICLE_SIZE_REFERENCE_PPU = 20.0f;
constexpr double BASE_STEPS_PER_SECOND = 60.0;
constexpr int MAX_STEPS_PER_FRAME = 256;

struct Viewport {
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;

  int fbX = 0;
  int fbY = 0;
  int fbWidth = 0;
  int fbHeight = 0;
};

struct Camera2D {
  float centerX = 0.0f;
  float centerY = 0.0f;
  float pixelsPerUnit = 20.0f;

  void fit_world(const WorldConfig &world, const Viewport &viewport) {
    centerX = 0.0f;
    centerY = 0.0f;

    if (viewport.width <= 0.0f || viewport.height <= 0.0f)
      return;

    const float sx = viewport.width / world.width;
    const float sy = viewport.height / world.height;
    pixelsPerUnit = std::clamp(0.92f * std::min(sx, sy), CAMERA_MIN_SCALE,
                               CAMERA_MAX_SCALE);
  }
};

struct ControlPanelState {
  SimulationConfig pending;

  bool panelOpen = true;
  float panelWidth = PANEL_DEFAULT_WIDTH;
  float panelLastWidth = PANEL_DEFAULT_WIDTH;
  float panelDragStartWidth = PANEL_DEFAULT_WIDTH;
  float panelDragStartMouseX = 0.0f;

  bool randomizePosition = true;
  bool randomizeVelocity = true;
  bool randomizeMass = true;
  bool randomizeInteraction = true;
  bool symmetricInteraction = false;

  float allParticleSize = 4.0f;
  float matrixLimit = 4.0f;

  int theme = 4;
  int regularFont = 0;
  int headerFont = 0;
  float fontSize = 24.0f;
  float uiScale = 1.2f;
  bool showStyleEditor = false;
};

struct FontEntry {
  std::string name;
  ImFont *font = nullptr;
};

struct GuiResources {
  std::vector<FontEntry> fonts;
};

struct GuiActions {
  bool apply = false;
  bool reinitialize = false;
  bool singleStep = false;
  bool fitWorld = false;
  bool centerView = false;
};

struct FrameStats {
  double simulationMs = 0.0; // CPU-side submission / execution time
  double gpuMs = -1.0;       // asynchronous GL_TIME_ELAPSED result
  int gpuTimedSteps = 0;
  int simulationSteps = 0;
};

struct GraphicsAdapterInfo {
  std::string name;
  std::uint64_t dedicatedMemory = 0;
  bool software = false;
  bool activeOpenGL = false;
};

struct OpenGLDeviceInfo {
  std::string vendor;
  std::string renderer;
  std::string version;
  int major = 0;
  int minor = 0;
  bool computeSupported = false;

  int maxWorkGroupInvocations = 0;
  int maxWorkGroupSizeX = 0;
  int maxWorkGroupCountX = 0;
  int maxSharedMemory = 0;
  int maxStorageBindings = 0;
  std::int64_t maxStorageBlockSize = 0;
};

struct HardwareInfo {
  int cpuThreads = 1;
  OpenGLDeviceInfo gl;
  std::vector<GraphicsAdapterInfo> graphicsAdapters;
  std::vector<CudaDeviceInfo> cudaDevices;
};

static int resolve_gpu_local_size(const HardwareInfo &hardware, int requested) {
  const int limit = std::max(1, std::min(hardware.gl.maxWorkGroupInvocations,
                                         hardware.gl.maxWorkGroupSizeX));

  if (requested > 0 && requested <= limit)
    return requested;

  // 256 is deliberately the aggressive automatic target. 512 threads per
  // group can reduce occupancy on real kernels even when the API permits it.
  constexpr int preferred[] = {256, 128, 64, 32, 16, 8, 4, 2, 1};
  for (int size : preferred)
    if (size <= limit)
      return size;
  return 1;
}

static int resolve_cuda_device(const HardwareInfo &hardware, int requested) {
  if (requested >= 0) {
    for (const CudaDeviceInfo &device : hardware.cudaDevices)
      if (device.id == requested && device.openGLCompatible)
        return device.id;
  }

  for (const CudaDeviceInfo &device : hardware.cudaDevices)
    if (device.openGLCompatible)
      return device.id;
  return -1;
}

static const CudaDeviceInfo *find_cuda_device(const HardwareInfo &hardware,
                                              int id) {
  for (const CudaDeviceInfo &device : hardware.cudaDevices)
    if (device.id == id)
      return &device;
  return nullptr;
}

static bool cuda_available(const HardwareInfo &hardware) {
  return CudaComputeBackend::compiled() &&
         resolve_cuda_device(hardware, -1) >= 0;
}

static std::string lower_ascii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

static int available_cpu_threads() {
#ifdef _OPENMP
  return std::max(1, omp_get_num_procs());
#else
  return std::max(1u, std::thread::hardware_concurrency());
#endif
}

static void apply_cpu_thread_count(int requested, int available) {
#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(requested > 0 ? std::clamp(requested, 1, available)
                                    : available);
#else
  (void)requested;
  (void)available;
#endif
}

static OpenGLDeviceInfo query_opengl_device() {
  OpenGLDeviceInfo info;

  const auto *vendor = glGetString(GL_VENDOR);
  const auto *renderer = glGetString(GL_RENDERER);
  const auto *version = glGetString(GL_VERSION);
  info.vendor = vendor ? reinterpret_cast<const char *>(vendor) : "Unknown";
  info.renderer =
      renderer ? reinterpret_cast<const char *>(renderer) : "Unknown";
  info.version = version ? reinterpret_cast<const char *>(version) : "Unknown";

  glGetIntegerv(GL_MAJOR_VERSION, &info.major);
  glGetIntegerv(GL_MINOR_VERSION, &info.minor);
  info.computeSupported =
      info.major > 4 || (info.major == 4 && info.minor >= 3);

  if (info.computeSupported) {
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
                  &info.maxWorkGroupInvocations);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &info.maxWorkGroupSizeX);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0,
                    &info.maxWorkGroupCountX);
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &info.maxSharedMemory);
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,
                  &info.maxStorageBindings);
    GLint64 storage = 0;
    glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &storage);
    info.maxStorageBlockSize = static_cast<std::int64_t>(storage);
  }

  return info;
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t *text) {
  if (!text || !*text)
    return {};
  const int count =
      WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
  if (count <= 1)
    return {};
  std::string result(static_cast<std::size_t>(count), '\0');
  WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), count, nullptr,
                      nullptr);
  result.pop_back();
  return result;
}
#endif

static std::vector<GraphicsAdapterInfo>
enumerate_graphics_adapters(const OpenGLDeviceInfo &glInfo) {
  std::vector<GraphicsAdapterInfo> adapters;

#ifdef _WIN32
  IDXGIFactory1 *factory = nullptr;
  const GUID factory1Iid = {0x770aae78,
                            0xf26f,
                            0x4dba,
                            {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
  if (SUCCEEDED(CreateDXGIFactory1(factory1Iid,
                                   reinterpret_cast<void **>(&factory)))) {
    for (UINT i = 0;; ++i) {
      IDXGIAdapter1 *adapter = nullptr;
      if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND)
        break;
      if (!adapter)
        continue;

      DXGI_ADAPTER_DESC1 desc{};
      if (SUCCEEDED(adapter->GetDesc1(&desc))) {
        GraphicsAdapterInfo item;
        item.name = wide_to_utf8(desc.Description);
        item.dedicatedMemory =
            static_cast<std::uint64_t>(desc.DedicatedVideoMemory);
        item.software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;

        const std::string nameLower = lower_ascii(item.name);
        const std::string rendererLower = lower_ascii(glInfo.renderer);
        item.activeOpenGL =
            !item.software &&
            (rendererLower.find(nameLower) != std::string::npos ||
             nameLower.find(rendererLower) != std::string::npos);
        adapters.push_back(std::move(item));
      }
      adapter->Release();
    }
    factory->Release();
  }
#elif defined(__linux__)
  const std::filesystem::path drmDir = "/sys/class/drm";
  std::error_code ec;
  if (std::filesystem::is_directory(drmDir, ec)) {
    for (const auto &entry : std::filesystem::directory_iterator(drmDir, ec)) {
      if (ec)
        break;
      const std::string card = entry.path().filename().string();
      if (card.size() < 5 || card.rfind("card", 0) != 0 ||
          !std::all_of(card.begin() + 4, card.end(),
                       [](unsigned char c) { return std::isdigit(c); }))
        continue;

      auto read_one_line = [](const std::filesystem::path &path) {
        std::ifstream file(path);
        std::string value;
        std::getline(file, value);
        return value;
      };

      const std::string vendorId =
          lower_ascii(read_one_line(entry.path() / "device/vendor"));
      const std::string deviceId =
          lower_ascii(read_one_line(entry.path() / "device/device"));

      std::string vendor = "GPU";
      if (vendorId == "0x10de")
        vendor = "NVIDIA";
      else if (vendorId == "0x1002")
        vendor = "AMD";
      else if (vendorId == "0x8086")
        vendor = "Intel";

      GraphicsAdapterInfo item;
      item.name = vendor + " " + card;
      if (!deviceId.empty())
        item.name += " (device " + deviceId + ")";

      const std::string vram =
          read_one_line(entry.path() / "device/mem_info_vram_total");
      if (!vram.empty()) {
        char *end = nullptr;
        const unsigned long long value = std::strtoull(vram.c_str(), &end, 10);
        if (end != vram.c_str())
          item.dedicatedMemory = static_cast<std::uint64_t>(value);
      }
      adapters.push_back(std::move(item));
    }
  }
#endif

  if (adapters.empty()) {
    GraphicsAdapterInfo current;
    current.name = glInfo.renderer;
    current.activeOpenGL = true;
    adapters.push_back(std::move(current));
  } else {
    bool foundActive = false;
    for (const auto &adapter : adapters)
      foundActive = foundActive || adapter.activeOpenGL;
    if (!foundActive) {
      GraphicsAdapterInfo current;
      current.name = glInfo.renderer + " (current OpenGL context)";
      current.activeOpenGL = true;
      adapters.push_back(std::move(current));
    }
  }

  return adapters;
}

static HardwareInfo query_hardware_info() {
  HardwareInfo info;
  info.cpuThreads = available_cpu_threads();
  info.gl = query_opengl_device();
  info.graphicsAdapters = enumerate_graphics_adapters(info.gl);
  info.cudaDevices = CudaComputeBackend::enumerate_devices();
  return info;
}

static GuiResources load_gui_fonts() {
  GuiResources resources;
  ImGuiIO &io = ImGui::GetIO();

  resources.fonts.push_back({"Default", io.Fonts->AddFontDefault()});

#ifdef PARTICLELIFE_EMBED_RESOURCES

  ImFontConfig regularConfig{};
  regularConfig.FontDataOwnedByAtlas = false;

  ImFont *regular = io.Fonts->AddFontFromMemoryTTF(
      EmbeddedResources::jetBrainsMonoRegular,
      static_cast<int>(sizeof(EmbeddedResources::jetBrainsMonoRegular)), 18.0f,
      &regularConfig);

  if (regular) {
    resources.fonts.push_back({"JetBrainsMono-Regular", regular});
  }

  ImFontConfig semiBoldConfig{};
  semiBoldConfig.FontDataOwnedByAtlas = false;

  ImFont *semiBold = io.Fonts->AddFontFromMemoryTTF(
      EmbeddedResources::jetBrainsMonoSemiBold,
      static_cast<int>(sizeof(EmbeddedResources::jetBrainsMonoSemiBold)), 18.0f,
      &semiBoldConfig);

  if (semiBold) {
    resources.fonts.push_back({"JetBrainsMono-SemiBold", semiBold});
  }

#else
  const std::filesystem::path fontDir = "resources/fonts";

  std::error_code ec;
  if (!std::filesystem::is_directory(fontDir, ec))
    return resources;

  std::vector<std::filesystem::path> paths;

  for (const auto &entry : std::filesystem::directory_iterator(fontDir, ec)) {
    if (ec)
      break;

    if (!entry.is_regular_file())
      continue;

    std::string ext = entry.path().extension().string();

    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    if (ext == ".ttf" || ext == ".otf")
      paths.push_back(entry.path());
  }

  std::sort(paths.begin(), paths.end());

  for (const auto &path : paths) {
    ImFont *font = io.Fonts->AddFontFromFileTTF(path.string().c_str(), 18.0f);

    if (font) {
      resources.fonts.push_back({path.stem().string(), font});
    }
  }

#endif

  return resources;
}

static std::string normalized_font_name(std::string name) {
  std::string result;
  result.reserve(name.size());
  for (unsigned char c : name) {
    if (std::isalnum(c))
      result.push_back(static_cast<char>(std::tolower(c)));
  }
  return result;
}

static int find_jetbrains_mono_font(const GuiResources &gui,
                                    const char *weight) {
  const std::string wantedWeight = normalized_font_name(weight);
  for (int i = 0; i < static_cast<int>(gui.fonts.size()); ++i) {
    const std::string name =
        normalized_font_name(gui.fonts[static_cast<std::size_t>(i)].name);
    const bool jetBrainsMono =
        name.find("jetbrainsmono") != std::string::npos ||
        name.find("jbmono") != std::string::npos;
    if (jetBrainsMono && name.find(wantedWeight) != std::string::npos)
      return i;
  }
  return 0;
}

static ImFont *selected_font(const GuiResources &gui, int index) {
  if (gui.fonts.empty())
    return nullptr;
  index = std::clamp(index, 0, static_cast<int>(gui.fonts.size()) - 1);
  return gui.fonts[static_cast<std::size_t>(index)].font;
}

static void apply_graphite_theme(ImGuiStyle &style) {
  ImGui::StyleColorsDark(&style);
  ImVec4 *c = style.Colors;
  c[ImGuiCol_WindowBg] = ImVec4(0.075f, 0.080f, 0.090f, 1.00f);
  c[ImGuiCol_ChildBg] = ImVec4(0.075f, 0.080f, 0.090f, 0.00f);
  c[ImGuiCol_FrameBg] = ImVec4(0.130f, 0.140f, 0.155f, 1.00f);
  c[ImGuiCol_FrameBgHovered] = ImVec4(0.180f, 0.195f, 0.215f, 1.00f);
  c[ImGuiCol_FrameBgActive] = ImVec4(0.220f, 0.235f, 0.260f, 1.00f);
  c[ImGuiCol_Header] = ImVec4(0.160f, 0.175f, 0.195f, 1.00f);
  c[ImGuiCol_HeaderHovered] = ImVec4(0.220f, 0.240f, 0.270f, 1.00f);
  c[ImGuiCol_HeaderActive] = ImVec4(0.260f, 0.285f, 0.320f, 1.00f);
  c[ImGuiCol_Button] = ImVec4(0.165f, 0.180f, 0.200f, 1.00f);
  c[ImGuiCol_ButtonHovered] = ImVec4(0.230f, 0.250f, 0.280f, 1.00f);
  c[ImGuiCol_ButtonActive] = ImVec4(0.280f, 0.305f, 0.345f, 1.00f);
  c[ImGuiCol_CheckMark] = ImVec4(0.430f, 0.700f, 1.000f, 1.00f);
  c[ImGuiCol_SliderGrab] = ImVec4(0.360f, 0.620f, 0.900f, 1.00f);
  c[ImGuiCol_SliderGrabActive] = ImVec4(0.480f, 0.760f, 1.000f, 1.00f);
  style.WindowRounding = 0.0f;
  style.FrameRounding = 4.0f;
  style.GrabRounding = 4.0f;
}

static void apply_midnight_theme(ImGuiStyle &style) {
  ImGui::StyleColorsDark(&style);
  ImVec4 *c = style.Colors;
  c[ImGuiCol_WindowBg] = ImVec4(0.035f, 0.045f, 0.070f, 1.00f);
  c[ImGuiCol_FrameBg] = ImVec4(0.075f, 0.095f, 0.140f, 1.00f);
  c[ImGuiCol_FrameBgHovered] = ImVec4(0.105f, 0.145f, 0.220f, 1.00f);
  c[ImGuiCol_FrameBgActive] = ImVec4(0.125f, 0.175f, 0.270f, 1.00f);
  c[ImGuiCol_Header] = ImVec4(0.085f, 0.125f, 0.195f, 1.00f);
  c[ImGuiCol_HeaderHovered] = ImVec4(0.115f, 0.175f, 0.285f, 1.00f);
  c[ImGuiCol_HeaderActive] = ImVec4(0.145f, 0.215f, 0.350f, 1.00f);
  c[ImGuiCol_Button] = ImVec4(0.090f, 0.135f, 0.210f, 1.00f);
  c[ImGuiCol_ButtonHovered] = ImVec4(0.125f, 0.190f, 0.310f, 1.00f);
  c[ImGuiCol_ButtonActive] = ImVec4(0.155f, 0.235f, 0.385f, 1.00f);
  c[ImGuiCol_CheckMark] = ImVec4(0.360f, 0.720f, 1.000f, 1.00f);
  c[ImGuiCol_SliderGrab] = ImVec4(0.300f, 0.600f, 0.950f, 1.00f);
  c[ImGuiCol_SliderGrabActive] = ImVec4(0.450f, 0.780f, 1.000f, 1.00f);
  style.WindowRounding = 0.0f;
  style.FrameRounding = 5.0f;
  style.GrabRounding = 5.0f;
}

static void apply_ui_style(const ControlPanelState &state) {
  ImGuiStyle &style = ImGui::GetStyle();
  style = ImGuiStyle{};

  switch (state.theme) {
  case 1:
    ImGui::StyleColorsLight(&style);
    break;
  case 2:
    ImGui::StyleColorsClassic(&style);
    break;
  case 3:
    apply_graphite_theme(style);
    break;
  case 4:
    apply_midnight_theme(style);
    break;
  default:
    ImGui::StyleColorsDark(&style);
    break;
  }

  style.WindowRounding = 0.0f;
  style.ScaleAllSizes(state.uiScale);
}

static bool section_header(const char *label, const ControlPanelState &state,
                           const GuiResources &gui, bool defaultOpen = true) {
  ImGui::PushFont(selected_font(gui, state.headerFont), state.fontSize * 1.14f);
  const bool open = ImGui::CollapsingHeader(
      label, defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : 0);
  ImGui::PopFont();
  return open;
}

static void numeric_item_hint() {
  if (!ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
    return;
  ImGui::BeginTooltip();
  ImGui::TextUnformatted("Drag to adjust");
  ImGui::TextDisabled("Double-click or Ctrl+click to type a value");
  ImGui::EndTooltip();
}

static bool edit_drag_float(const char *label, float *value, float speed,
                            float minValue = 0.0f, float maxValue = 0.0f,
                            const char *format = "%.3f") {
  const bool changed = ImGui::DragFloat(label, value, speed, minValue, maxValue,
                                        format, ImGuiSliderFlags_AlwaysClamp);
  numeric_item_hint();
  return changed;
}

static bool
edit_drag_float_range(const char *label, float *minValue, float *maxValue,
                      float speed,
                      float lowerBound = -std::numeric_limits<float>::max(),
                      float upperBound = std::numeric_limits<float>::max(),
                      const char *format = "%.3f") {
  const bool changed = ImGui::DragFloatRange2(
      label, minValue, maxValue, speed, lowerBound, upperBound, format, format,
      ImGuiSliderFlags_AlwaysClamp);
  if (*minValue > *maxValue)
    std::swap(*minValue, *maxValue);
  numeric_item_hint();
  return changed;
}

static bool edit_drag_int(const char *label, int *value, float speed,
                          int minValue = 0, int maxValue = 0) {
  const bool changed = ImGui::DragInt(label, value, speed, minValue, maxValue,
                                      "%d", ImGuiSliderFlags_AlwaysClamp);
  numeric_item_hint();
  return changed;
}

static bool edit_slider_float(const char *label, float *value, float minValue,
                              float maxValue, const char *format = "%.3f") {
  const bool changed = ImGui::SliderFloat(label, value, minValue, maxValue,
                                          format, ImGuiSliderFlags_AlwaysClamp);
  numeric_item_hint();
  return changed;
}

static bool edit_slider_int(const char *label, int *value, int minValue,
                            int maxValue) {
  const bool changed = ImGui::SliderInt(label, value, minValue, maxValue, "%d",
                                        ImGuiSliderFlags_AlwaysClamp);
  numeric_item_hint();
  return changed;
}

static std::string load_text_file(const char *path) {
#ifdef PARTICLELIFE_EMBED_RESOURCES

  std::string source = EmbeddedResources::load_text(path);

  if (source.empty())
    std::cerr << "Embedded resource not found: " << path << '\n';

  return source;

#else

  std::ifstream file(path);
  if (!file) {
    std::cerr << "Cannot open file: " << path << '\n';
    return {};
  }

  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();

#endif
}

static GLuint compile_shader(GLenum type, const std::string &source) {
  GLuint shader = glCreateShader(type);
  const char *src = source.c_str();

  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char log[2048];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    std::cerr << "Shader compile error:\n" << log << '\n';
    glDeleteShader(shader);
    return 0;
  }

  return shader;
}

static GLuint load_shader_program(const char *vertexPath,
                                  const char *fragmentPath) {
  const std::string vertexSource = load_text_file(vertexPath);
  const std::string fragmentSource = load_text_file(fragmentPath);
  if (vertexSource.empty() || fragmentSource.empty())
    return 0;

  GLuint vertexShader = compile_shader(GL_VERTEX_SHADER, vertexSource);
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
    std::cerr << "Shader link error:\n" << log << '\n';
    glDeleteProgram(program);
    return 0;
  }

  return program;
}

class ParticleRenderer {
public:
  bool initialize(const ParticleSet &particles) {
    program_ = load_shader_program("resources/shaders/particle.vert",
                                   "resources/shaders/particle.frag");
    compositeProgram_ =
        load_shader_program("resources/shaders/particle_composite.vert",
                            "resources/shaders/particle_composite.frag");
    glowExtractProgram_ =
        load_shader_program("resources/shaders/particle_composite.vert",
                            "resources/shaders/particle_glow_extract.frag");
    glowBlurProgram_ =
        load_shader_program("resources/shaders/particle_composite.vert",
                            "resources/shaders/particle_glow_blur.frag");

    if (!program_ || !compositeProgram_ || !glowExtractProgram_ ||
        !glowBlurProgram_)
      return false;

    centerLocation_ = glGetUniformLocation(program_, "uCenter");
    halfSizeLocation_ = glGetUniformLocation(program_, "uHalfSize");
    pointScaleLocation_ = glGetUniformLocation(program_, "uPointScale");
    particleSizeLocation_ = glGetUniformLocation(program_, "uParticleSize[0]");
    particleColorLocation_ =
        glGetUniformLocation(program_, "uParticleColor[0]");

    compositeTextureLocation_ =
        glGetUniformLocation(compositeProgram_, "uAccumTexture");
    compositeGlowTextureLocation_ =
        glGetUniformLocation(compositeProgram_, "uGlowTexture");
    compositeGlowEnabledLocation_ =
        glGetUniformLocation(compositeProgram_, "uGlowEnabled");
    compositeGlowStrengthLocation_ =
        glGetUniformLocation(compositeProgram_, "uGlowStrength");
    compositeGlowExposureLocation_ =
        glGetUniformLocation(compositeProgram_, "uGlowExposure");

    glowExtractTextureLocation_ =
        glGetUniformLocation(glowExtractProgram_, "uAccumTexture");
    glowExtractDensityLocation_ =
        glGetUniformLocation(glowExtractProgram_, "uGlowDensity");

    glowBlurTextureLocation_ =
        glGetUniformLocation(glowBlurProgram_, "uInputTexture");
    glowBlurDirectionLocation_ =
        glGetUniformLocation(glowBlurProgram_, "uDirection");
    glowBlurRadiusLocation_ = glGetUniformLocation(glowBlurProgram_, "uRadius");

    glGenVertexArrays(1, &vao_);
    glGenVertexArrays(1, &gpuVao_);
    glGenVertexArrays(1, &compositeVao_);
    glGenBuffers(1, &vboX_);
    glGenBuffers(1, &vboY_);
    glGenBuffers(1, &vboType_);

    glGenFramebuffers(1, &accumFramebuffer_);
    glGenTextures(1, &accumTexture_);
    glGenFramebuffers(2, glowFramebuffers_.data());
    glGenTextures(2, glowTextures_.data());

    glBindVertexArray(vao_);

    glBindBuffer(GL_ARRAY_BUFFER, vboX_);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vboY_);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, vboType_);
    glVertexAttribIPointer(2, 1, GL_UNSIGNED_SHORT, 0, nullptr);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    glUseProgram(compositeProgram_);
    glUniform1i(compositeTextureLocation_, 0);
    glUniform1i(compositeGlowTextureLocation_, 1);

    glUseProgram(glowExtractProgram_);
    glUniform1i(glowExtractTextureLocation_, 0);

    glUseProgram(glowBlurProgram_);
    glUniform1i(glowBlurTextureLocation_, 0);
    glUseProgram(0);

    rebuild(particles);
    return true;
  }

  void rebuild(const ParticleSet &particles) {
    const GLsizeiptr realBytes =
        static_cast<GLsizeiptr>(particles.particle_count() * sizeof(Real_t));
    const GLsizeiptr typeBytes = static_cast<GLsizeiptr>(
        particles.particle_count() * sizeof(ParticleType_t));

    glBindBuffer(GL_ARRAY_BUFFER, vboX_);
    glBufferData(GL_ARRAY_BUFFER, realBytes, particles.x_data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vboY_);
    glBufferData(GL_ARRAY_BUFFER, realBytes, particles.y_data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vboType_);
    glBufferData(GL_ARRAY_BUFFER, typeBytes, particles.type_data(),
                 GL_DYNAMIC_DRAW);
  }

  void render(const ParticleSet &particles, const SimulationConfig &config,
              const Camera2D &camera, const Viewport &viewport,
              float framebufferScaleY) {
    if (viewport.fbWidth <= 0 || viewport.fbHeight <= 0 ||
        viewport.width <= 0.0f || viewport.height <= 0.0f)
      return;

    upload_particle_data(particles);
    if (!begin_accumulation(config, camera, viewport, framebufferScaleY))
      return;

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0,
                 static_cast<GLsizei>(particles.particle_count()));
    glBindVertexArray(0);

    const bool glowReady = render_glow(config, viewport);
    composite(config, viewport, glowReady);
  }

  void render_gpu(GLuint particleStateBuffer, GLuint particleTypeBuffer,
                  const ParticleSet &particles, const SimulationConfig &config,
                  const Camera2D &camera, const Viewport &viewport,
                  float framebufferScaleY) {
    if (!particleStateBuffer || !particleTypeBuffer || viewport.fbWidth <= 0 ||
        viewport.fbHeight <= 0 || viewport.width <= 0.0f ||
        viewport.height <= 0.0f)
      return;

    if (!begin_accumulation(config, camera, viewport, framebufferScaleY))
      return;

    glBindVertexArray(gpuVao_);
    if (gpuVaoParticleBuffer_ != particleStateBuffer) {
      glBindBuffer(GL_ARRAY_BUFFER, particleStateBuffer);
      glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(Real_t),
                            reinterpret_cast<void *>(0));
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(Real_t),
                            reinterpret_cast<void *>(sizeof(Real_t)));
      glEnableVertexAttribArray(1);
      gpuVaoParticleBuffer_ = particleStateBuffer;
    }

    if (gpuVaoTypeBuffer_ != particleTypeBuffer) {
      glBindBuffer(GL_ARRAY_BUFFER, particleTypeBuffer);
      glVertexAttribIPointer(2, 1, GL_UNSIGNED_INT, sizeof(std::uint32_t),
                             nullptr);
      glEnableVertexAttribArray(2);
      gpuVaoTypeBuffer_ = particleTypeBuffer;
    }

    glDrawArrays(GL_POINTS, 0,
                 static_cast<GLsizei>(particles.particle_count()));
    glBindVertexArray(0);

    const bool glowReady = render_glow(config, viewport);
    composite(config, viewport, glowReady);
  }

  void destroy() {
    if (vboX_)
      glDeleteBuffers(1, &vboX_);
    if (vboY_)
      glDeleteBuffers(1, &vboY_);
    if (vboType_)
      glDeleteBuffers(1, &vboType_);
    if (vao_)
      glDeleteVertexArrays(1, &vao_);
    if (gpuVao_)
      glDeleteVertexArrays(1, &gpuVao_);
    if (compositeVao_)
      glDeleteVertexArrays(1, &compositeVao_);
    if (accumTexture_)
      glDeleteTextures(1, &accumTexture_);
    if (accumFramebuffer_)
      glDeleteFramebuffers(1, &accumFramebuffer_);
    glDeleteTextures(2, glowTextures_.data());
    glDeleteFramebuffers(2, glowFramebuffers_.data());
    if (program_)
      glDeleteProgram(program_);
    if (compositeProgram_)
      glDeleteProgram(compositeProgram_);
    if (glowExtractProgram_)
      glDeleteProgram(glowExtractProgram_);
    if (glowBlurProgram_)
      glDeleteProgram(glowBlurProgram_);

    vboX_ = vboY_ = vboType_ = 0;
    vao_ = gpuVao_ = compositeVao_ = 0;
    accumTexture_ = accumFramebuffer_ = 0;
    glowTextures_.fill(0);
    glowFramebuffers_.fill(0);
    gpuVaoParticleBuffer_ = 0;
    gpuVaoTypeBuffer_ = 0;
    program_ = compositeProgram_ = 0;
    glowExtractProgram_ = glowBlurProgram_ = 0;
    accumWidth_ = accumHeight_ = 0;
    glowWidth_ = glowHeight_ = 0;
  }

private:
  bool ensure_accumulation_target(int width, int height) {
    if (width == accumWidth_ && height == accumHeight_)
      return true;

    glBindTexture(GL_TEXTURE_2D, accumTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA,
                 GL_FLOAT, nullptr);

    glBindFramebuffer(GL_FRAMEBUFFER, accumFramebuffer_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           accumTexture_, 0);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (status != GL_FRAMEBUFFER_COMPLETE) {
      std::cerr << "Particle accumulation framebuffer incomplete: 0x"
                << std::hex << status << std::dec << '\n';
      accumWidth_ = accumHeight_ = 0;
      return false;
    }

    accumWidth_ = width;
    accumHeight_ = height;
    return true;
  }

  bool ensure_glow_targets(int width, int height) {
    // Half-resolution bloom is intentionally fixed: it is considerably
    // cheaper and the lower resolution actually helps create a soft halo.
    width = std::max(1, (width + 1) / 2);
    height = std::max(1, (height + 1) / 2);
    if (width == glowWidth_ && height == glowHeight_)
      return true;

    for (int i = 0; i < 2; ++i) {
      glBindTexture(GL_TEXTURE_2D, glowTextures_[i]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA,
                   GL_FLOAT, nullptr);

      glBindFramebuffer(GL_FRAMEBUFFER, glowFramebuffers_[i]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, glowTextures_[i], 0);
      const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
      if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Particle glow framebuffer incomplete: 0x" << std::hex
                  << status << std::dec << '\n';
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glowWidth_ = glowHeight_ = 0;
        return false;
      }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glowWidth_ = width;
    glowHeight_ = height;
    return true;
  }

  bool begin_accumulation(const SimulationConfig &config,
                          const Camera2D &camera, const Viewport &viewport,
                          float framebufferScaleY) {
    if (!ensure_accumulation_target(viewport.fbWidth, viewport.fbHeight))
      return false;

    const float viewHalfWidth = 0.5f * viewport.width / camera.pixelsPerUnit;
    const float viewHalfHeight = 0.5f * viewport.height / camera.pixelsPerUnit;

    glBindFramebuffer(GL_FRAMEBUFFER, accumFramebuffer_);
    glViewport(0, 0, viewport.fbWidth, viewport.fbHeight);
    glDisable(GL_SCISSOR_TEST);

    constexpr GLfloat clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearBufferfv(GL_COLOR, 0, clear);

    // Accumulate premultiplied color and total particle coverage. Addition
    // is commutative, so GPU particle reordering cannot change the image.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    glUseProgram(program_);
    glUniform2f(centerLocation_, camera.centerX, camera.centerY);
    glUniform2f(halfSizeLocation_, viewHalfWidth, viewHalfHeight);
    glUniform1f(pointScaleLocation_,
                (camera.pixelsPerUnit / PARTICLE_SIZE_REFERENCE_PPU) *
                    framebufferScaleY);

    std::array<float, 16> sizes{};
    std::array<float, 16 * 4> colors{};
    for (Index_t i = 0; i < config.particleTypes.size() && i < 16; ++i) {
      sizes[i] = config.particleTypes[i].size;
      const Color &c = config.particleTypes[i].color;
      colors[4 * i + 0] = c.r;
      colors[4 * i + 1] = c.g;
      colors[4 * i + 2] = c.b;
      colors[4 * i + 3] = c.a;
    }

    glUniform1fv(particleSizeLocation_, 16, sizes.data());
    glUniform4fv(particleColorLocation_, 16, colors.data());
    return true;
  }

  bool render_glow(const SimulationConfig &config, const Viewport &viewport) {
    if (!config.display.glowEnabled)
      return false;
    if (!ensure_glow_targets(viewport.fbWidth, viewport.fbHeight))
      return false;

    glDisable(GL_BLEND);
    glBindVertexArray(compositeVao_);
    glViewport(0, 0, glowWidth_, glowHeight_);

    // Downsample the order-independent accumulation texture and convert
    // local density into an emission image.
    glBindFramebuffer(GL_FRAMEBUFFER, glowFramebuffers_[0]);
    glUseProgram(glowExtractProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, accumTexture_);
    glUniform1f(glowExtractDensityLocation_, config.display.glowDensity);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Horizontal blur: glow A -> glow B.
    glBindFramebuffer(GL_FRAMEBUFFER, glowFramebuffers_[1]);
    glUseProgram(glowBlurProgram_);
    glBindTexture(GL_TEXTURE_2D, glowTextures_[0]);
    glUniform2f(glowBlurDirectionLocation_, 1.0f, 0.0f);
    glUniform1f(glowBlurRadiusLocation_, config.display.glowRadius);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Vertical blur: glow B -> glow A. glowTextures_[0] is therefore the
    // final bloom texture sampled by the composite pass.
    glBindFramebuffer(GL_FRAMEBUFFER, glowFramebuffers_[0]);
    glBindTexture(GL_TEXTURE_2D, glowTextures_[1]);
    glUniform2f(glowBlurDirectionLocation_, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
  }

  void composite(const SimulationConfig &config, const Viewport &viewport,
                 bool glowReady) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(viewport.fbX, viewport.fbY, viewport.fbWidth, viewport.fbHeight);
    glEnable(GL_SCISSOR_TEST);
    glScissor(viewport.fbX, viewport.fbY, viewport.fbWidth, viewport.fbHeight);

    glDisable(GL_BLEND);
    glUseProgram(compositeProgram_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, accumTexture_);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, glowReady ? glowTextures_[0] : 0);

    glUniform1i(compositeGlowEnabledLocation_, glowReady ? 1 : 0);
    glUniform1f(compositeGlowStrengthLocation_, config.display.glowStrength);
    glUniform1f(compositeGlowExposureLocation_, config.display.glowExposure);

    glBindVertexArray(compositeVao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Restore the ordinary UI-friendly blend state. ImGui also sets its
    // own state, but keeping a sane global state avoids surprises.
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_SCISSOR_TEST);
  }

  void upload_particle_data(const ParticleSet &particles) {
    const GLsizeiptr realBytes =
        static_cast<GLsizeiptr>(particles.particle_count() * sizeof(Real_t));
    const GLsizeiptr typeBytes = static_cast<GLsizeiptr>(
        particles.particle_count() * sizeof(ParticleType_t));

    glBindBuffer(GL_ARRAY_BUFFER, vboX_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, realBytes, particles.x_data());

    glBindBuffer(GL_ARRAY_BUFFER, vboY_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, realBytes, particles.y_data());

    glBindBuffer(GL_ARRAY_BUFFER, vboType_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, typeBytes, particles.type_data());
  }

private:
  GLuint program_ = 0;
  GLuint compositeProgram_ = 0;
  GLuint glowExtractProgram_ = 0;
  GLuint glowBlurProgram_ = 0;

  GLuint vao_ = 0;
  GLuint gpuVao_ = 0;
  GLuint compositeVao_ = 0;
  GLuint vboX_ = 0;
  GLuint vboY_ = 0;
  GLuint vboType_ = 0;

  GLuint accumFramebuffer_ = 0;
  GLuint accumTexture_ = 0;
  std::array<GLuint, 2> glowFramebuffers_{};
  std::array<GLuint, 2> glowTextures_{};

  GLuint gpuVaoParticleBuffer_ = 0;
  GLuint gpuVaoTypeBuffer_ = 0;

  int accumWidth_ = 0;
  int accumHeight_ = 0;
  int glowWidth_ = 0;
  int glowHeight_ = 0;

  GLint centerLocation_ = -1;
  GLint halfSizeLocation_ = -1;
  GLint pointScaleLocation_ = -1;
  GLint particleSizeLocation_ = -1;
  GLint particleColorLocation_ = -1;

  GLint compositeTextureLocation_ = -1;
  GLint compositeGlowTextureLocation_ = -1;
  GLint compositeGlowEnabledLocation_ = -1;
  GLint compositeGlowStrengthLocation_ = -1;
  GLint compositeGlowExposureLocation_ = -1;

  GLint glowExtractTextureLocation_ = -1;
  GLint glowExtractDensityLocation_ = -1;

  GLint glowBlurTextureLocation_ = -1;
  GLint glowBlurDirectionLocation_ = -1;
  GLint glowBlurRadiusLocation_ = -1;
};

static Viewport make_viewport(float panelWidth, int windowWidth,
                              int windowHeight, int framebufferWidth,
                              int framebufferHeight) {
  Viewport viewport;

  panelWidth = std::clamp(panelWidth, 0.0f, static_cast<float>(windowWidth));
  viewport.x = panelWidth;
  viewport.y = 0.0f;
  viewport.width = std::max(0.0f, static_cast<float>(windowWidth) - panelWidth);
  viewport.height = static_cast<float>(windowHeight);

  const float scaleX = windowWidth > 0 ? static_cast<float>(framebufferWidth) /
                                             static_cast<float>(windowWidth)
                                       : 1.0f;
  const float scaleY = windowHeight > 0
                           ? static_cast<float>(framebufferHeight) /
                                 static_cast<float>(windowHeight)
                           : 1.0f;

  viewport.fbX = static_cast<int>(std::lround(viewport.x * scaleX));
  viewport.fbY = 0;
  viewport.fbWidth = std::max(0, framebufferWidth - viewport.fbX);
  viewport.fbHeight = framebufferHeight;

  return viewport;
}

static bool mouse_inside(const Viewport &viewport, const ImVec2 &mouse) {
  return mouse.x >= viewport.x && mouse.x < viewport.x + viewport.width &&
         mouse.y >= viewport.y && mouse.y < viewport.y + viewport.height;
}

static void handle_camera_input(Camera2D &camera, const Viewport &viewport) {
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureMouse || !mouse_inside(viewport, io.MousePos))
    return;

  if (io.MouseWheel != 0.0f) {
    const float mouseOffsetX =
        io.MousePos.x - (viewport.x + 0.5f * viewport.width);
    const float mouseOffsetY =
        io.MousePos.y - (viewport.y + 0.5f * viewport.height);

    const float worldX = camera.centerX + mouseOffsetX / camera.pixelsPerUnit;
    const float worldY = camera.centerY - mouseOffsetY / camera.pixelsPerUnit;

    const float factor = std::pow(1.15f, io.MouseWheel);
    camera.pixelsPerUnit = std::clamp(camera.pixelsPerUnit * factor,
                                      CAMERA_MIN_SCALE, CAMERA_MAX_SCALE);

    camera.centerX = worldX - mouseOffsetX / camera.pixelsPerUnit;
    camera.centerY = worldY + mouseOffsetY / camera.pixelsPerUnit;
  }

  if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
      ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
    camera.centerX -= io.MouseDelta.x / camera.pixelsPerUnit;
    camera.centerY += io.MouseDelta.y / camera.pixelsPerUnit;
  }
}

static ImVec2 world_to_screen(float x, float y, const Camera2D &camera,
                              const Viewport &viewport) {
  return {
      viewport.x + 0.5f * viewport.width +
          (x - camera.centerX) * camera.pixelsPerUnit,
      viewport.y + 0.5f * viewport.height -
          (y - camera.centerY) * camera.pixelsPerUnit,
  };
}

static void draw_world_overlay(const SimulationConfig &config,
                               const Camera2D &camera,
                               const Viewport &viewport) {
  if (viewport.width <= 0.0f || viewport.height <= 0.0f)
    return;

  ImDrawList *draw = ImGui::GetBackgroundDrawList();
  draw->PushClipRect(
      {viewport.x, viewport.y},
      {viewport.x + viewport.width, viewport.y + viewport.height}, true);

  const float halfWidth = 0.5f * config.world.width;
  const float halfHeight = 0.5f * config.world.height;

  if (config.display.showGrid &&
      config.world.bucketSize * camera.pixelsPerUnit >= 3.0f) {
    const Color &c = config.display.gridColor;
    const ImU32 gridColor = ImGui::ColorConvertFloat4ToU32(ImVec4(
        c.r, c.g, c.b, std::clamp(config.display.gridOpacity, 0.0f, 1.0f)));

    for (float x = -halfWidth; x <= halfWidth + 1.0e-5f;
         x += config.world.bucketSize) {
      const ImVec2 a = world_to_screen(x, -halfHeight, camera, viewport);
      const ImVec2 b = world_to_screen(x, halfHeight, camera, viewport);
      draw->AddLine(a, b, gridColor, config.display.gridThickness);
    }

    for (float y = -halfHeight; y <= halfHeight + 1.0e-5f;
         y += config.world.bucketSize) {
      const ImVec2 a = world_to_screen(-halfWidth, y, camera, viewport);
      const ImVec2 b = world_to_screen(halfWidth, y, camera, viewport);
      draw->AddLine(a, b, gridColor, config.display.gridThickness);
    }
  }

  const ImVec2 min = world_to_screen(-halfWidth, halfHeight, camera, viewport);
  const ImVec2 max = world_to_screen(halfWidth, -halfHeight, camera, viewport);
  draw->AddRect(min, max, IM_COL32(220, 220, 220, 115), 0.0f, 0, 1.5f);

  draw->PopClipRect();
}

static float snap_world_size(float size, float bucketSize) {
  const float cells = std::max(1.0f, std::round(size / bucketSize));
  return cells * bucketSize;
}

static Color default_color(Index_t index) {
  constexpr Color colors[] = {
      {1.00f, 0.25f, 0.25f, 0.70f}, {0.25f, 0.65f, 1.00f, 0.70f},
      {0.25f, 1.00f, 0.40f, 0.70f}, {1.00f, 0.75f, 0.20f, 0.70f},
      {0.80f, 0.35f, 1.00f, 0.70f}, {0.20f, 1.00f, 0.90f, 0.70f},
      {1.00f, 0.35f, 0.70f, 0.70f}, {0.75f, 0.80f, 1.00f, 0.70f},
  };
  return colors[index % (sizeof(colors) / sizeof(colors[0]))];
}

static void resize_types(SimulationConfig &config, Index_t count) {
  const Index_t oldCount = config.particleTypes.size();
  config.particleTypes.resize(count);
  for (Index_t i = oldCount; i < count; ++i)
    config.particleTypes[i].color = default_color(i);
  config.interaction.resize(count);
}

static ImU32 matrix_color(float value, float limit) {
  const float t =
      std::clamp(std::abs(value) / std::max(limit, 1.0e-6f), 0.0f, 1.0f);

  if (value > 0.0f)
    return ImGui::ColorConvertFloat4ToU32(ImVec4(
        0.20f + 0.75f * t, 0.18f * (1.0f - t), 0.18f * (1.0f - t), 1.0f));

  if (value < 0.0f)
    return ImGui::ColorConvertFloat4ToU32(
        ImVec4(0.18f * (1.0f - t), 0.22f + 0.25f * t, 0.25f + 0.70f * t, 1.0f));

  return IM_COL32(52, 52, 56, 255);
}

static void symmetrize_matrix(SquareMatrix &matrix) {
  for (Index_t i = 0; i < matrix.width; ++i) {
    for (Index_t j = i + 1; j < matrix.width; ++j) {
      const float value = 0.5f * (matrix(i, j) + matrix(j, i));
      matrix(i, j) = value;
      matrix(j, i) = value;
    }
  }
}

static void mirror_upper_triangle(SquareMatrix &matrix) {
  for (Index_t i = 0; i < matrix.width; ++i)
    for (Index_t j = i + 1; j < matrix.width; ++j)
      matrix(j, i) = matrix(i, j);
}

static void draw_interaction_matrix(SquareMatrix &matrix, float limit,
                                    bool symmetric) {
  const float cellSize = std::max(18.0f, ImGui::GetFrameHeight() * 1.30f);
  const float gap = std::max(2.0f, ImGui::GetStyle().ItemSpacing.x * 0.35f);
  const float labelWidth = cellSize;

  ImGui::TextUnformatted("Rows <- columns");
  ImGui::TextDisabled("Drag: adjust   Double-click: type   Right click: zero");

  const float childHeight = std::min(
      430.0f,
      labelWidth + static_cast<float>(matrix.width) * (cellSize + gap) + 18.0f);

  ImGui::BeginChild("##interaction_matrix", ImVec2(0.0f, childHeight),
                    ImGuiChildFlags_Borders,
                    ImGuiWindowFlags_HorizontalScrollbar);

  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList *draw = ImGui::GetWindowDrawList();

  for (Index_t j = 0; j < matrix.width; ++j) {
    const float x =
        origin.x + labelWidth + static_cast<float>(j) * (cellSize + gap);
    draw->AddText(ImVec2(x + 7.0f, origin.y), ImGui::GetColorU32(ImGuiCol_Text),
                  std::to_string(j).c_str());
  }

  ImGui::SetCursorScreenPos({origin.x, origin.y + labelWidth});

  for (Index_t i = 0; i < matrix.width; ++i) {
    const float y =
        origin.y + labelWidth + static_cast<float>(i) * (cellSize + gap);
    draw->AddText(ImVec2(origin.x + 4.0f, y + 5.0f),
                  ImGui::GetColorU32(ImGuiCol_Text), std::to_string(i).c_str());

    for (Index_t j = 0; j < matrix.width; ++j) {
      const float x =
          origin.x + labelWidth + static_cast<float>(j) * (cellSize + gap);

      ImGui::SetCursorScreenPos({x, y});
      ImGui::PushID(static_cast<int>(i * matrix.width + j));
      ImGui::InvisibleButton("##cell", {cellSize, cellSize},
                             ImGuiButtonFlags_MouseButtonLeft |
                                 ImGuiButtonFlags_MouseButtonRight);

      const ImVec2 min = ImGui::GetItemRectMin();
      const ImVec2 max = ImGui::GetItemRectMax();
      const bool cellHovered = ImGui::IsItemHovered();

      float &value = matrix(i, j);
      if (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const float fine = ImGui::GetIO().KeyShift ? 0.1f : 1.0f;
        value += ImGui::GetIO().MouseDelta.x * limit * 0.01f * fine;
        value = std::clamp(value, -limit, limit);
        if (symmetric && i != j)
          matrix(j, i) = value;
      }

      if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        value = 0.0f;
        if (symmetric && i != j)
          matrix(j, i) = 0.0f;
      }

      if (cellHovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        ImGui::OpenPopup("##matrix_value");

      if (ImGui::BeginPopup("##matrix_value")) {
        ImGui::Text("Type %zu <- Type %zu", i, j);
        ImGui::SetNextItemWidth(130.0f);
        if (ImGui::InputFloat("Value", &value, 0.0f, 0.0f, "%+.4f",
                              ImGuiInputTextFlags_EnterReturnsTrue)) {
          value = std::clamp(value, -limit, limit);
          if (symmetric && i != j)
            matrix(j, i) = value;
          ImGui::CloseCurrentPopup();
        }
        value = std::clamp(value, -limit, limit);
        if (symmetric && i != j)
          matrix(j, i) = value;
        ImGui::EndPopup();
      }

      draw->AddRectFilled(min, max, matrix_color(value, limit), 3.0f);

      if (cellHovered) {
        draw->AddRect(min, max, ImGui::GetColorU32(ImGuiCol_Text), 3.0f, 0,
                      1.5f);
        ImGui::BeginTooltip();
        ImGui::Text("Type %zu <- Type %zu", i, j);
        ImGui::Text("%+.4f", value);
        ImGui::TextDisabled("+ repulsive, - attractive");
        ImGui::EndTooltip();
      }

      ImGui::PopID();
    }
  }

  ImGui::SetCursorScreenPos(
      {origin.x, origin.y + labelWidth +
                     static_cast<float>(matrix.width) * (cellSize + gap) +
                     4.0f});
  ImGui::Dummy(
      {labelWidth + static_cast<float>(matrix.width) * (cellSize + gap), 1.0f});

  ImGui::EndChild();
}

static bool topology_changed(const SimulationConfig &a,
                             const SimulationConfig &b) {
  if (a.particleTypes.size() != b.particleTypes.size())
    return true;

  for (Index_t i = 0; i < a.particleTypes.size(); ++i)
    if (a.particleTypes[i].count != b.particleTypes[i].count)
      return true;

  return false;
}

static bool grid_changed(const SimulationConfig &a, const SimulationConfig &b) {
  return a.world.width != b.world.width || a.world.height != b.world.height ||
         a.world.bucketSize != b.world.bucketSize ||
         a.world.boundary != b.world.boundary;
}

static bool font_combo(const char *label, int &current,
                       const GuiResources &gui) {
  if (gui.fonts.empty())
    return false;

  current = std::clamp(current, 0, static_cast<int>(gui.fonts.size()) - 1);
  bool changed = false;
  if (ImGui::BeginCombo(
          label, gui.fonts[static_cast<std::size_t>(current)].name.c_str())) {
    for (int i = 0; i < static_cast<int>(gui.fonts.size()); ++i) {
      const bool selected = i == current;
      if (ImGui::Selectable(gui.fonts[static_cast<std::size_t>(i)].name.c_str(),
                            selected)) {
        current = i;
        changed = true;
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  return changed;
}

static void draw_panel_resize_handle(ControlPanelState &state, int windowWidth,
                                     int windowHeight) {
  const float maxWidth = std::max(
      PANEL_MIN_WIDTH,
      std::min(PANEL_MAX_WIDTH, static_cast<float>(windowWidth) * 0.80f));

  if (!state.panelOpen) {
    ImGui::SetNextWindowPos({0.0f, 0.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize(
        {PANEL_HANDLE_WIDTH + 4.0f, static_cast<float>(windowHeight)},
        ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::Begin("##panel_open_handle", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoBackground);

    const float cy = 0.5f * static_cast<float>(windowHeight);
    ImGui::SetCursorScreenPos({0.0f, cy - 25.0f});
    ImGui::InvisibleButton("##open", {PANEL_HANDLE_WIDTH + 4.0f, 50.0f});
    if (ImGui::IsItemHovered())
      ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    if (ImGui::IsItemClicked()) {
      state.panelOpen = true;
      state.panelWidth =
          std::clamp(state.panelLastWidth, PANEL_MIN_WIDTH, maxWidth);
    }

    ImDrawList *draw = ImGui::GetWindowDrawList();
    const ImU32 bg = ImGui::GetColorU32(ImGuiCol_Button);
    const ImU32 fg = ImGui::GetColorU32(ImGuiCol_Text);
    draw->AddRectFilled({0.0f, cy - 25.0f},
                        {PANEL_HANDLE_WIDTH + 4.0f, cy + 25.0f}, bg, 3.0f);
    draw->AddText({3.0f, cy - 0.5f * ImGui::GetFontSize()}, fg, ">");
    ImGui::End();
    return;
  }

  state.panelWidth = std::clamp(state.panelWidth, PANEL_MIN_WIDTH, maxWidth);

  const float x = state.panelWidth - 0.5f * PANEL_HANDLE_WIDTH;
  ImGui::SetNextWindowPos({x, 0.0f}, ImGuiCond_Always);
  ImGui::SetNextWindowSize(
      {PANEL_HANDLE_WIDTH, static_cast<float>(windowHeight)}, ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.0f);
  ImGui::Begin("##panel_resize_handle", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
                   ImGuiWindowFlags_NoBackground);

  ImGui::SetCursorScreenPos({x, 0.0f});
  ImGui::InvisibleButton(
      "##resize", {PANEL_HANDLE_WIDTH, static_cast<float>(windowHeight)});

  if (ImGui::IsItemActivated()) {
    state.panelDragStartWidth = state.panelWidth;
    state.panelDragStartMouseX = ImGui::GetIO().MousePos.x;
  }

  if (ImGui::IsItemActive()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    const float drag = ImGui::GetIO().MousePos.x - state.panelDragStartMouseX;
    const float requested = state.panelDragStartWidth + drag;

    if (requested <= PANEL_COLLAPSE_THRESHOLD) {
      state.panelLastWidth = state.panelDragStartWidth;
      state.panelOpen = false;
    } else {
      state.panelWidth = std::clamp(requested, PANEL_MIN_WIDTH, maxWidth);
      state.panelLastWidth = state.panelWidth;
    }
  } else if (ImGui::IsItemHovered()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
  }

  const float cy = 0.5f * static_cast<float>(windowHeight);
  if (ImGui::IsItemDeactivated()) {
    const float moved =
        std::abs(ImGui::GetIO().MousePos.x - state.panelDragStartMouseX);
    const bool onToggle = std::abs(ImGui::GetIO().MousePos.y - cy) <= 25.0f;
    if (moved < 3.0f && onToggle) {
      state.panelLastWidth = state.panelWidth;
      state.panelOpen = false;
    }
  }

  ImDrawList *draw = ImGui::GetWindowDrawList();
  const ImU32 line =
      ImGui::GetColorU32((ImGui::IsItemHovered() || ImGui::IsItemActive())
                             ? ImGuiCol_SliderGrabActive
                             : ImGuiCol_Border);
  const ImU32 bg = ImGui::GetColorU32(ImGuiCol_Button);
  const ImU32 fg = ImGui::GetColorU32(ImGuiCol_Text);
  draw->AddLine(
      {x + 0.5f * PANEL_HANDLE_WIDTH, 0.0f},
      {x + 0.5f * PANEL_HANDLE_WIDTH, static_cast<float>(windowHeight)}, line,
      1.0f);
  draw->AddRectFilled({x, cy - 25.0f}, {x + PANEL_HANDLE_WIDTH, cy + 25.0f}, bg,
                      3.0f);
  draw->AddText({x + 3.0f, cy - 0.5f * ImGui::GetFontSize()}, fg, "<");
  ImGui::End();
}

static std::string format_bytes(std::uint64_t bytes) {
  constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
  constexpr double MiB = 1024.0 * 1024.0;
  char buffer[64];
  if (bytes >= static_cast<std::uint64_t>(GiB))
    std::snprintf(buffer, sizeof(buffer), "%.2f GiB", bytes / GiB);
  else
    std::snprintf(buffer, sizeof(buffer), "%.0f MiB", bytes / MiB);
  return buffer;
}

static GuiActions
draw_control_panel(ControlPanelState &state, const GuiResources &gui,
                   Simulation &simulation, Camera2D &camera,
                   const Viewport &viewport, const FrameStats &stats,
                   const HardwareInfo &hardware, bool openGLBackendReady,
                   bool cudaBackendReady, int windowWidth, int windowHeight) {
  GuiActions actions;
  SimulationConfig &active = simulation.config();
  SimulationConfig &pending = state.pending;

  if (!state.panelOpen)
    return actions;

  const float maxWidth = std::max(
      PANEL_MIN_WIDTH,
      std::min(PANEL_MAX_WIDTH, static_cast<float>(windowWidth) * 0.80f));
  state.panelWidth = std::clamp(state.panelWidth, PANEL_MIN_WIDTH, maxWidth);

  ImGui::SetNextWindowPos({0.0f, 0.0f}, ImGuiCond_Always);
  ImGui::SetNextWindowSize({state.panelWidth, static_cast<float>(windowHeight)},
                           ImGuiCond_Always);
  ImGui::Begin("Particle Life", nullptr,
               ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoSavedSettings);

  ImGui::TextDisabled("Numbers: drag, or double-click / Ctrl+click to type");

  if (section_header("Random / Initialization", state, gui)) {
    ImGui::InputScalar("Seed", ImGuiDataType_U32, &pending.random.seed);
    ImGui::SameLine();
    if (ImGui::Button("New"))
      pending.random.seed = std::random_device{}();

    ImGui::Checkbox("Position", &state.randomizePosition);
    ImGui::SameLine();
    ImGui::Checkbox("Velocity", &state.randomizeVelocity);
    ImGui::Checkbox("Mass", &state.randomizeMass);
    ImGui::SameLine();
    ImGui::Checkbox("Interaction##RandomizeInteraction",
                    &state.randomizeInteraction);

    const float halfPendingWidth = 0.5f * pending.world.width;
    const float halfPendingHeight = 0.5f * pending.world.height;

    edit_drag_float_range("Position X", &pending.random.posMinX,
                          &pending.random.posMaxX, 0.1f, -halfPendingWidth,
                          halfPendingWidth);
    edit_drag_float_range("Position Y", &pending.random.posMinY,
                          &pending.random.posMaxY, 0.1f, -halfPendingHeight,
                          halfPendingHeight);

    if (ImGui::Button("Use Current World")) {
      pending.random.posMinX = -0.5f * pending.world.width;
      pending.random.posMaxX = 0.5f * pending.world.width;
      pending.random.posMinY = -0.5f * pending.world.height;
      pending.random.posMaxY = 0.5f * pending.world.height;
    }

    edit_drag_float_range("Velocity X", &pending.random.velMinX,
                          &pending.random.velMaxX, 0.01f);
    edit_drag_float_range("Velocity Y", &pending.random.velMinY,
                          &pending.random.velMaxY, 0.01f);
    edit_drag_float_range("Mass Range", &pending.random.massMin,
                          &pending.random.massMax, 0.05f, 0.001f, 100.0f);
    edit_drag_float_range("Matrix Range", &pending.random.interactionMin,
                          &pending.random.interactionMax, 0.05f);

    if (ImGui::Button("Reinitialize"))
      actions.reinitialize = true;
  }

  if (section_header("Time", state, gui)) {
    edit_drag_float("Time Step", &pending.time.dt, 0.0001f, 0.00001f, 0.1f,
                    "%.5f");
    edit_slider_float("Speed", &active.time.speed, 0.0f, 10.0f, "%.2fx");
    ImGui::Checkbox("Pause", &active.time.paused);
    ImGui::SameLine();
    if (ImGui::Button("Step"))
      actions.singleStep = true;
  }

  if (section_header("World", state, gui)) {
    int width = static_cast<int>(std::lround(pending.world.width));
    int height = static_cast<int>(std::lround(pending.world.height));
    if (edit_drag_int("Width", &width, 1.0f, 8, 512))
      pending.world.width =
          snap_world_size(static_cast<float>(width), pending.world.bucketSize);
    if (edit_drag_int("Height", &height, 1.0f, 8, 512))
      pending.world.height =
          snap_world_size(static_cast<float>(height), pending.world.bucketSize);

    const char *boundaryItems[] = {"Periodic", "Reflective"};
    int boundary = pending.world.boundary == BoundaryMode::Periodic ? 0 : 1;
    if (ImGui::Combo("Boundary", &boundary, boundaryItems, 2))
      pending.world.boundary =
          boundary == 0 ? BoundaryMode::Periodic : BoundaryMode::Reflective;

    if (pending.world.boundary == BoundaryMode::Reflective)
      edit_slider_float("Restitution", &pending.world.restitution, 0.0f, 1.0f,
                        "%.3f");

    constexpr float bucketSizes[] = {0.25f, 0.5f, 1.0f, 2.0f};
    const char *bucketLabels[] = {"0.25", "0.5", "1.0", "2.0"};
    int bucketIndex = 2;
    for (int i = 0; i < 4; ++i)
      if (pending.world.bucketSize == bucketSizes[i])
        bucketIndex = i;
    if (ImGui::Combo("Bucket Size", &bucketIndex, bucketLabels, 4)) {
      pending.world.bucketSize = bucketSizes[bucketIndex];
      pending.world.width =
          snap_world_size(pending.world.width, pending.world.bucketSize);
      pending.world.height =
          snap_world_size(pending.world.height, pending.world.bucketSize);
    }
  }

  if (section_header("Particles", state, gui)) {
    int typeCount = static_cast<int>(pending.particleTypes.size());
    if (edit_slider_int("Type Count", &typeCount, 1, 16)) {
      const Index_t oldCount = pending.particleTypes.size();
      resize_types(pending, static_cast<Index_t>(typeCount));
      for (Index_t i = oldCount; i < pending.particleTypes.size(); ++i)
        pending.particleTypes[i].size = state.allParticleSize;
    }

    if (edit_drag_float("All Type Sizes", &state.allParticleSize, 0.1f, 1.0f,
                        64.0f, "%.2f")) {
      for (ParticleTypeConfig &type : pending.particleTypes)
        type.size = state.allParticleSize;
    }
    ImGui::TextDisabled(
        "Changing this value sets every type to the same size.");
    ImGui::TextDisabled(
        "Particle size scales with camera zoom like world geometry.");

    for (Index_t i = 0; i < pending.particleTypes.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      const std::string label = "Type " + std::to_string(i);
      if (ImGui::TreeNode(label.c_str())) {
        ParticleTypeConfig &type = pending.particleTypes[i];

        int count = static_cast<int>(std::min<Index_t>(type.count, 1000000));
        if (edit_drag_int("Count", &count, 100.0f, 1, 1000000))
          type.count = static_cast<Index_t>(count);

        edit_drag_float("Mass", &type.mass, 0.01f, 0.001f, 100.0f);
        edit_drag_float("Size", &type.size, 0.1f, 1.0f, 64.0f);
        ImGui::ColorEdit4("Color", &type.color.r, ImGuiColorEditFlags_AlphaBar);

        ImGui::TreePop();
      }
      ImGui::PopID();
    }
  }

  if (section_header("Interaction", state, gui)) {
    edit_slider_float("Core Radius", &pending.physics.coreRadius, 0.01f, 0.99f,
                      "%.3f");
    edit_drag_float("Core Strength", &pending.physics.coreStrength, 0.05f, 0.0f,
                    100.0f);
    edit_drag_float("Global Damping", &pending.physics.damping, 0.05f, 0.0f,
                    100.0f);
    edit_drag_float("Core Damping", &pending.physics.localDamping, 0.05f, 0.0f,
                    100.0f);

    edit_slider_float("Matrix Display Limit", &state.matrixLimit, 0.25f, 16.0f,
                      "%.2f");

    if (ImGui::Checkbox("Symmetric Matrix", &state.symmetricInteraction) &&
        state.symmetricInteraction) {
      symmetrize_matrix(pending.interaction);
    }
    ImGui::TextDisabled("When enabled, editing M[i][j] also updates M[j][i].");

    draw_interaction_matrix(pending.interaction, state.matrixLimit,
                            state.symmetricInteraction);

    if (ImGui::Button("Zero Matrix"))
      std::fill(pending.interaction.data.begin(),
                pending.interaction.data.end(), 0.0f);
  }

  if (section_header("Compute", state, gui)) {
    const bool openGLAvailable =
        hardware.gl.computeSupported && openGLBackendReady;
    const bool cudaUsable = cuda_available(hardware);

    const char *currentBackend = "CPU / OpenMP";
    if (pending.compute.backend == ComputeBackend::OpenGLCompute)
      currentBackend = "GPU / OpenGL Compute";
    else if (pending.compute.backend == ComputeBackend::Cuda)
      currentBackend = "GPU / CUDA";

    if (ImGui::BeginCombo("Backend", currentBackend)) {
      const bool cpuSelected =
          pending.compute.backend == ComputeBackend::CpuOpenMP;
      if (ImGui::Selectable("CPU / OpenMP", cpuSelected))
        pending.compute.backend = ComputeBackend::CpuOpenMP;
      if (cpuSelected)
        ImGui::SetItemDefaultFocus();

      ImGui::BeginDisabled(!openGLAvailable);
      const bool glSelected =
          pending.compute.backend == ComputeBackend::OpenGLCompute;
      if (ImGui::Selectable("GPU / OpenGL Compute", glSelected))
        pending.compute.backend = ComputeBackend::OpenGLCompute;
      ImGui::EndDisabled();

      ImGui::BeginDisabled(!cudaUsable);
      const bool cudaSelected = pending.compute.backend == ComputeBackend::Cuda;
      if (ImGui::Selectable("GPU / CUDA", cudaSelected))
        pending.compute.backend = ComputeBackend::Cuda;
      ImGui::EndDisabled();
      ImGui::EndCombo();
    }

    if (!hardware.gl.computeSupported)
      ImGui::TextDisabled(
          "OpenGL Compute unavailable: OpenGL 4.3+ is required.");
    else if (!openGLBackendReady)
      ImGui::TextDisabled("OpenGL Compute shader failed to initialize.");

    if (!CudaComputeBackend::compiled())
      ImGui::TextDisabled("CUDA backend not compiled into this build.");
    else if (!cudaUsable)
      ImGui::TextDisabled(
          "CUDA: no device compatible with the current OpenGL context.");
    else if (!cudaBackendReady)
      ImGui::TextDisabled(
          "CUDA backend initialization failed; Apply can retry.");

    if (pending.compute.backend == ComputeBackend::CpuOpenMP) {
      bool automatic = pending.compute.cpuThreads == 0;
      if (ImGui::Checkbox("Automatic Threads", &automatic))
        pending.compute.cpuThreads = automatic ? 0 : hardware.cpuThreads;

      if (!automatic)
        edit_slider_int("Threads", &pending.compute.cpuThreads, 1,
                        hardware.cpuThreads);

#ifdef _OPENMP
      ImGui::Text("OpenMP processors: %d", hardware.cpuThreads);
#else
      ImGui::Text("Hardware threads: %d", hardware.cpuThreads);
      ImGui::TextDisabled(
          "This build has no OpenMP support; CPU backend is serial.");
#endif
    } else if (pending.compute.backend == ComputeBackend::OpenGLCompute) {
      ImGui::Text("Active device: %s", hardware.gl.renderer.c_str());
      ImGui::Text("Vendor: %s", hardware.gl.vendor.c_str());
      ImGui::Text("OpenGL: %d.%d", hardware.gl.major, hardware.gl.minor);
      ImGui::Text("Work-group invocations: %d",
                  hardware.gl.maxWorkGroupInvocations);
      ImGui::Text("Max work-group X: %d", hardware.gl.maxWorkGroupSizeX);
      ImGui::Text("Shared memory: %.1f KiB",
                  hardware.gl.maxSharedMemory / 1024.0f);

      const int resolvedLocalSize =
          resolve_gpu_local_size(hardware, pending.compute.gpuWorkGroupSize);
      const char *localSizeLabel =
          pending.compute.gpuWorkGroupSize == 0
              ? "Auto (recommended)"
              : (pending.compute.gpuWorkGroupSize == 64    ? "64"
                 : pending.compute.gpuWorkGroupSize == 128 ? "128"
                 : pending.compute.gpuWorkGroupSize == 256 ? "256"
                                                           : "512");

      if (ImGui::BeginCombo("Work Group Size", localSizeLabel)) {
        const bool autoSelected = pending.compute.gpuWorkGroupSize == 0;
        if (ImGui::Selectable("Auto (recommended)", autoSelected))
          pending.compute.gpuWorkGroupSize = 0;
        if (autoSelected)
          ImGui::SetItemDefaultFocus();

        for (int option : OpenGLComputeBackend::LOCAL_SIZE_OPTIONS) {
          const bool supported =
              option <= hardware.gl.maxWorkGroupInvocations &&
              option <= hardware.gl.maxWorkGroupSizeX;
          ImGui::BeginDisabled(!supported);
          const std::string label = std::to_string(option);
          const bool selected = pending.compute.gpuWorkGroupSize == option;
          if (ImGui::Selectable(label.c_str(), selected) && supported)
            pending.compute.gpuWorkGroupSize = option;
          ImGui::EndDisabled();
        }
        ImGui::EndCombo();
      }
      ImGui::TextDisabled("Resolved local size after Apply: %d",
                          resolvedLocalSize);
      ImGui::TextDisabled("OpenGL backend remains the cross-vendor GPU path.");
    } else {
      const int resolvedDevice =
          resolve_cuda_device(hardware, pending.compute.cudaDevice);
      const CudaDeviceInfo *currentCuda =
          find_cuda_device(hardware, resolvedDevice);
      const char *cudaLabel =
          currentCuda ? currentCuda->name.c_str() : "Unavailable";

      if (ImGui::BeginCombo("CUDA Device", cudaLabel)) {
        for (const CudaDeviceInfo &device : hardware.cudaDevices) {
          ImGui::BeginDisabled(!device.openGLCompatible);
          const bool selected = device.id == resolvedDevice;
          std::string label =
              device.name + "##cuda" + std::to_string(device.id);
          if (ImGui::Selectable(label.c_str(), selected) &&
              device.openGLCompatible)
            pending.compute.cudaDevice = device.id;
          if (selected)
            ImGui::SetItemDefaultFocus();
          ImGui::EndDisabled();
        }
        ImGui::EndCombo();
      }

      if (currentCuda) {
        ImGui::Text("Compute capability: %d.%d", currentCuda->major,
                    currentCuda->minor);
        ImGui::Text("VRAM: %s", format_bytes(currentCuda->totalMemory).c_str());
        ImGui::Text("SM count: %d", currentCuda->multiprocessorCount);
        ImGui::Text("Warp size: %d", currentCuda->warpSize);
        ImGui::Text("Max threads / block: %d", currentCuda->maxThreadsPerBlock);
        ImGui::Text("OpenGL interop: %s",
                    currentCuda->openGLCompatible ? "yes" : "no");
      }
      ImGui::TextDisabled("CUDA optimized path: CUB scan + physical reorder.");
      ImGui::TextDisabled(
          "Dense same-cell blocks use shared-memory neighbor tiling.");
      ImGui::TextDisabled(
          "Launch sizes are selected from actual kernel occupancy.");
    }

    ImGui::SeparatorText("Graphics Adapters");
    for (const GraphicsAdapterInfo &adapter : hardware.graphicsAdapters) {
      if (adapter.activeOpenGL)
        ImGui::Text("* %s  [OpenGL context]", adapter.name.c_str());
      else
        ImGui::Text("  %s", adapter.name.c_str());

      if (adapter.dedicatedMemory > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s",
                            format_bytes(adapter.dedicatedMemory).c_str());
      }
      if (adapter.software) {
        ImGui::SameLine();
        ImGui::TextDisabled("software");
      }
    }

    if (!hardware.cudaDevices.empty()) {
      ImGui::SeparatorText("CUDA Devices");
      for (const CudaDeviceInfo &device : hardware.cudaDevices) {
        ImGui::Text("%c GPU %d: %s", device.openGLCompatible ? '*' : ' ',
                    device.id, device.name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("%s  CC %d.%d",
                            format_bytes(device.totalMemory).c_str(),
                            device.major, device.minor);
      }
      ImGui::TextDisabled("* = compatible with the current OpenGL context");
    }
  }

  if (section_header("View / Performance", state, gui)) {
    bool vsync = active.display.vsync;
    if (ImGui::Checkbox("VSync", &vsync)) {
      active.display.vsync = vsync;
      glfwSwapInterval(vsync ? 1 : 0);
    }

    ImGui::Checkbox("Show Grid", &active.display.showGrid);
    if (active.display.showGrid) {
      edit_slider_float("Grid Opacity", &active.display.gridOpacity, 0.0f, 1.0f,
                        "%.2f");
      edit_drag_float("Grid Thickness", &active.display.gridThickness, 0.05f,
                      0.5f, 4.0f, "%.2f px");
      ImGui::ColorEdit3("Grid Color", &active.display.gridColor.r);
    }

    ImGui::SeparatorText("Fluorescent Glow");
    ImGui::Checkbox("Glow Enabled", &active.display.glowEnabled);
    if (active.display.glowEnabled) {
      edit_slider_float("Glow Strength", &active.display.glowStrength, 0.0f,
                        3.0f, "%.2f");
      edit_slider_float("Glow Radius", &active.display.glowRadius, 0.25f, 4.0f,
                        "%.2f");
      edit_slider_float("Density Response", &active.display.glowDensity, 0.25f,
                        3.0f, "%.2f");
      edit_slider_float("Glow Exposure", &active.display.glowExposure, 0.25f,
                        3.0f, "%.2f");
      ImGui::TextDisabled(
          "Half-resolution separable bloom; order-independent.");
      ImGui::TextDisabled(
          "Lower Density Response makes isolated particles glow more.");
    }

    edit_drag_int("FPS Limit", &active.display.fpsLimit, 1.0f, 0, 1000);
    ImGui::TextDisabled("0 = unlimited (when VSync is off)");

    ImGui::SeparatorText("Camera");
    edit_drag_float("Center X", &camera.centerX,
                    0.05f / std::max(camera.pixelsPerUnit, 1.0f));
    edit_drag_float("Center Y", &camera.centerY,
                    0.05f / std::max(camera.pixelsPerUnit, 1.0f));
    edit_drag_float("Scale (px/unit)", &camera.pixelsPerUnit, 0.2f,
                    CAMERA_MIN_SCALE, CAMERA_MAX_SCALE, "%.2f");
    camera.pixelsPerUnit =
        std::clamp(camera.pixelsPerUnit, CAMERA_MIN_SCALE, CAMERA_MAX_SCALE);

    if (ImGui::Button("Fit World"))
      actions.fitWorld = true;
    ImGui::SameLine();
    if (ImGui::Button("Center"))
      actions.centerView = true;

    ImGui::TextDisabled("Wheel: zoom around cursor");
    ImGui::TextDisabled("Middle / right drag: pan");

    ImGui::SeparatorText("Statistics");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    if (active.compute.backend == ComputeBackend::OpenGLCompute ||
        active.compute.backend == ComputeBackend::Cuda) {
      ImGui::Text("CPU submit: %.3f ms", stats.simulationMs);
      if (stats.gpuMs >= 0.0) {
        ImGui::Text("GPU batch: %.3f ms", stats.gpuMs);
        if (stats.gpuTimedSteps > 0)
          ImGui::Text("GPU / step: %.3f ms", stats.gpuMs / stats.gpuTimedSteps);
      } else {
        ImGui::TextDisabled("GPU timer warming up...");
      }
    } else {
      ImGui::Text("Simulation: %.3f ms", stats.simulationMs);
    }
    ImGui::Text("Steps / frame: %d", stats.simulationSteps);
    ImGui::Text("Particles: %zu", simulation.particles().particle_count());
  }

  if (section_header("Interface", state, gui, false)) {
    const char *themes[] = {"Dark", "Light", "Classic", "Graphite", "Midnight"};
    int theme = state.theme;
    if (ImGui::Combo("Theme", &theme, themes, 5)) {
      state.theme = theme;
      apply_ui_style(state);
    }

    font_combo("Font", state.regularFont, gui);
    font_combo("Header Font", state.headerFont, gui);

    edit_drag_float("Font Size", &state.fontSize, 0.25f, 10.0f, 40.0f,
                    "%.1f px");

    float uiScale = state.uiScale;
    if (edit_drag_float("UI Scale", &uiScale, 0.01f, 0.70f, 2.00f, "%.2fx")) {
      state.uiScale = uiScale;
      apply_ui_style(state);
    }

    float panelWidth = state.panelWidth;
    if (edit_drag_float("Panel Width", &panelWidth, 1.0f, PANEL_MIN_WIDTH,
                        maxWidth, "%.0f px")) {
      state.panelWidth = panelWidth;
      state.panelLastWidth = panelWidth;
    }

    ImGui::Checkbox("Show ImGui Style Editor", &state.showStyleEditor);
#ifdef PARTICLELIFE_EMBED_RESOURCES
    ImGui::TextDisabled("Fonts are embedded in this executable.");
#else
    ImGui::TextDisabled("Fonts are scanned from resources/fonts at startup.");

    ImGui::TextDisabled(
        "Use any .ttf or .otf file; restart after adding fonts.");
#endif
  }

  ImGui::Separator();
  if (ImGui::Button("Apply Parameters", {-1.0f, 34.0f}))
    actions.apply = true;

  ImGui::End();

  if (state.showStyleEditor)
    ImGui::ShowStyleEditor();

  return actions;
}

static void download_active_backend_state(Simulation &simulation,
                                          OpenGLComputeBackend &openGL,
                                          CudaComputeBackend &cuda) {
  const ComputeBackend backend = simulation.config().compute.backend;
  if (backend == ComputeBackend::OpenGLCompute && openGL.valid())
    openGL.download_state(simulation);
  else if (backend == ComputeBackend::Cuda && cuda.valid())
    cuda.download_state(simulation);
}

static void
apply_pending_config(Simulation &simulation, ParticleRenderer &renderer,
                     OpenGLComputeBackend &openGL, CudaComputeBackend &cuda,
                     const HardwareInfo &hardware, ControlPanelState &panel) {
  SimulationConfig &active = simulation.config();
  SimulationConfig &pending = panel.pending;

  // Synchronize the currently authoritative accelerator state to CPU only
  // when Apply actually requests a backend/configuration transition.
  download_active_backend_state(simulation, openGL, cuda);

  const bool topologyChanged = topology_changed(active, pending);
  const bool gridChanged = grid_changed(active, pending);

  active.random = pending.random;
  active.time.dt = pending.time.dt;
  active.world = pending.world;
  active.physics = pending.physics;
  active.particleTypes = pending.particleTypes;
  active.interaction = pending.interaction;
  active.compute = pending.compute;

  if (active.compute.backend == ComputeBackend::OpenGLCompute &&
      !openGL.valid())
    active.compute.backend = ComputeBackend::CpuOpenMP;

  if (active.compute.backend == ComputeBackend::Cuda &&
      !cuda_available(hardware))
    active.compute.backend = ComputeBackend::CpuOpenMP;

  const bool useAccelerator =
      active.compute.backend == ComputeBackend::OpenGLCompute ||
      active.compute.backend == ComputeBackend::Cuda;

  if (topologyChanged) {
    simulation.rebuild();
    simulation.randomize_particles(!useAccelerator);
    renderer.rebuild(simulation.particles());
  } else if (gridChanged) {
    simulation.rebuild_grid(!useAccelerator);
  } else if (!useAccelerator) {
    simulation.refresh_acceleration();
  }

  apply_cpu_thread_count(active.compute.cpuThreads, hardware.cpuThreads);

  if (active.compute.backend == ComputeBackend::OpenGLCompute) {
    const int localSize =
        resolve_gpu_local_size(hardware, active.compute.gpuWorkGroupSize);
    if (!openGL.set_local_size(static_cast<GLuint>(localSize), active)) {
      active.compute.backend = ComputeBackend::CpuOpenMP;
      simulation.refresh_acceleration();
    } else {
      openGL.rebuild(simulation);
    }
  } else if (active.compute.backend == ComputeBackend::Cuda) {
    const int device = resolve_cuda_device(hardware, active.compute.cudaDevice);
    if (!cuda.valid() || cuda.device_id() != device) {
      cuda.destroy();
      if (!cuda.initialize(simulation, device)) {
        active.compute.backend = ComputeBackend::CpuOpenMP;
        simulation.refresh_acceleration();
      }
    } else {
      cuda.rebuild(simulation);
      if (!cuda.valid()) {
        active.compute.backend = ComputeBackend::CpuOpenMP;
        simulation.refresh_acceleration();
      }
    }
    if (active.compute.backend == ComputeBackend::Cuda)
      active.compute.cudaDevice = cuda.device_id();
  }

  panel.pending = active;
}

static GLFWwindow *create_main_window() {
  auto try_create = [](int major, int minor) {
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    return glfwCreateWindow(1200, 900, "VParticleLife", nullptr, nullptr);
  };

  // Prefer 4.3 because it guarantees compute shaders and SSBOs.
  if (GLFWwindow *window = try_create(4, 3))
    return window;

  // Old GPUs/drivers still get the complete CPU backend.
  return try_create(3, 3);
}

} // namespace

static int particlelife_main() {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW\n";
    return 1;
  }

  GLFWwindow *window = create_main_window();
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

  const HardwareInfo hardware = query_hardware_info();
  std::cout << "OpenGL: " << hardware.gl.version << '\n';
  std::cout << "Renderer: " << hardware.gl.renderer << '\n';
  std::cout << "OpenGL Compute: "
            << (hardware.gl.computeSupported ? "available" : "unavailable")
            << '\n';
  std::cout << "CUDA backend: "
            << (CudaComputeBackend::compiled() ? "compiled" : "not compiled")
            << '\n';
  for (const CudaDeviceInfo &device : hardware.cudaDevices) {
    std::cout << "CUDA GPU " << device.id << ": " << device.name << "  CC "
              << device.major << '.' << device.minor
              << (device.openGLCompatible ? "  [OpenGL interop]" : "") << '\n';
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  // VParticleLife does not persist ImGui window/layout state.
  // Setting this to nullptr disables both loading and saving imgui.ini.
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;

  GuiResources gui = load_gui_fonts();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  Simulation simulation(make_default_config());
  ParticleRenderer renderer;
  if (!renderer.initialize(simulation.particles())) {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
  }

  apply_cpu_thread_count(simulation.config().compute.cpuThreads,
                         hardware.cpuThreads);

  OpenGLComputeBackend gpuBackend;
  bool openGLBackendReady = false;
  if (hardware.gl.computeSupported && hardware.gl.maxStorageBindings >= 8) {
    const int localSize = resolve_gpu_local_size(
        hardware, simulation.config().compute.gpuWorkGroupSize);
    openGLBackendReady =
        gpuBackend.initialize(simulation, static_cast<GLuint>(localSize));
  }

  CudaComputeBackend cudaBackend;
  bool cudaBackendReady = false;
  if (cuda_available(hardware))
    cudaBackendReady = cudaBackend.initialize(simulation, -1);

  // Prefer CUDA on compatible NVIDIA systems, otherwise keep the universal
  // OpenGL Compute path, and finally fall back to CPU/OpenMP.
  if (cudaBackendReady) {
    simulation.config().compute.backend = ComputeBackend::Cuda;
    simulation.config().compute.cudaDevice = cudaBackend.device_id();
  } else if (openGLBackendReady) {
    simulation.config().compute.backend = ComputeBackend::OpenGLCompute;
  } else {
    simulation.config().compute.backend = ComputeBackend::CpuOpenMP;
  }

  ControlPanelState panel{simulation.config()};
  if (!panel.pending.particleTypes.empty())
    panel.allParticleSize = panel.pending.particleTypes.front().size;

  panel.regularFont = find_jetbrains_mono_font(gui, "Regular");
  panel.headerFont = find_jetbrains_mono_font(gui, "SemiBold");
  apply_ui_style(panel);

  Camera2D camera;
  FrameStats stats;

  glfwSwapInterval(simulation.config().display.vsync ? 1 : 0);

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);

  int windowWidth = 0;
  int windowHeight = 0;
  int framebufferWidth = 0;
  int framebufferHeight = 0;
  glfwGetWindowSize(window, &windowWidth, &windowHeight);
  glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
  camera.fit_world(simulation.config().world,
                   make_viewport(panel.panelOpen ? panel.panelWidth : 0.0f,
                                 windowWidth, windowHeight, framebufferWidth,
                                 framebufferHeight));

  double previousTime = glfwGetTime();
  double stepAccumulator = 0.0;

  while (!glfwWindowShouldClose(window)) {
    const auto frameStart = std::chrono::steady_clock::now();

    glfwPollEvents();

    const double now = glfwGetTime();
    const double realDelta = std::min(now - previousTime, 0.25);
    previousTime = now;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

    ImGui::PushFont(selected_font(gui, panel.regularFont), panel.fontSize);

    Viewport viewport =
        make_viewport(panel.panelOpen ? panel.panelWidth : 0.0f, windowWidth,
                      windowHeight, framebufferWidth, framebufferHeight);

    const GuiActions actions = draw_control_panel(
        panel, gui, simulation, camera, viewport, stats, hardware,
        openGLBackendReady, cudaBackendReady, windowWidth, windowHeight);
    draw_panel_resize_handle(panel, windowWidth, windowHeight);

    // The panel width/open state may have changed this frame.
    viewport =
        make_viewport(panel.panelOpen ? panel.panelWidth : 0.0f, windowWidth,
                      windowHeight, framebufferWidth, framebufferHeight);

    if (actions.apply) {
      apply_pending_config(simulation, renderer, gpuBackend, cudaBackend,
                           hardware, panel);
      openGLBackendReady = gpuBackend.valid();
      cudaBackendReady = cudaBackend.valid();
    }

    if (actions.reinitialize) {
      download_active_backend_state(simulation, gpuBackend, cudaBackend);

      const bool useOpenGL = simulation.config().compute.backend ==
                                 ComputeBackend::OpenGLCompute &&
                             gpuBackend.valid();
      const bool useCuda =
          simulation.config().compute.backend == ComputeBackend::Cuda &&
          cudaBackend.valid();
      const bool useAccelerator = useOpenGL || useCuda;

      simulation.config().random = panel.pending.random;
      simulation.randomize_selected(
          panel.randomizePosition, panel.randomizeVelocity, panel.randomizeMass,
          panel.randomizeInteraction, !useAccelerator);

      if (panel.randomizeInteraction && panel.symmetricInteraction) {
        // Keep the random distribution uniform: generate the upper
        // triangle once, then mirror it instead of averaging two
        // independent random values.
        mirror_upper_triangle(simulation.config().interaction);
        if (!useAccelerator)
          simulation.refresh_acceleration();
      }

      if (panel.randomizeMass) {
        const Index_t count =
            std::min(panel.pending.particleTypes.size(),
                     simulation.config().particleTypes.size());
        for (Index_t i = 0; i < count; ++i)
          panel.pending.particleTypes[i].mass =
              simulation.config().particleTypes[i].mass;
      }

      if (panel.randomizeInteraction) {
        const Index_t count = std::min(panel.pending.interaction.width,
                                       simulation.config().interaction.width);
        for (Index_t i = 0; i < count; ++i)
          for (Index_t j = 0; j < count; ++j)
            panel.pending.interaction(i, j) =
                simulation.config().interaction(i, j);
      }

      if (useOpenGL)
        gpuBackend.upload_state(simulation);
      else if (useCuda)
        cudaBackend.upload_state(simulation);
    }

    if (actions.fitWorld)
      camera.fit_world(simulation.config().world, viewport);
    if (actions.centerView) {
      camera.centerX = 0.0f;
      camera.centerY = 0.0f;
    }

    handle_camera_input(camera, viewport);

    stats.simulationSteps = 0;
    const auto simStart = std::chrono::steady_clock::now();

    const bool useOpenGLNow =
        simulation.config().compute.backend == ComputeBackend::OpenGLCompute &&
        gpuBackend.valid();
    const bool useCudaNow =
        simulation.config().compute.backend == ComputeBackend::Cuda &&
        cudaBackend.valid();
    if (actions.singleStep) {
      if (useOpenGLNow)
        gpuBackend.step(simulation.config(), simulation.config().time.dt);
      else if (useCudaNow)
        cudaBackend.step(simulation.config(), simulation.config().time.dt);
      else
        simulation.step();
      stats.simulationSteps = 1;
      stepAccumulator = 0.0;
    } else if (!simulation.config().time.paused) {
      stepAccumulator +=
          realDelta * BASE_STEPS_PER_SECOND * simulation.config().time.speed;

      const int requested = static_cast<int>(stepAccumulator);
      const int steps = std::min(requested, MAX_STEPS_PER_FRAME);

      if (steps > 0) {
        if (useOpenGLNow) {
          gpuBackend.step_many(simulation.config(), simulation.config().time.dt,
                               steps);
        } else if (useCudaNow) {
          cudaBackend.step_many(simulation.config(),
                                simulation.config().time.dt, steps);
        } else {
          for (int i = 0; i < steps; ++i)
            simulation.step();
        }
      }

      stepAccumulator -= static_cast<double>(steps);
      if (requested > MAX_STEPS_PER_FRAME)
        stepAccumulator = std::fmod(stepAccumulator, 1.0);

      stats.simulationSteps = steps;
    } else {
      stepAccumulator = 0.0;
    }

    const auto simEnd = std::chrono::steady_clock::now();
    stats.simulationMs =
        std::chrono::duration<double, std::milli>(simEnd - simStart).count();
    if (useOpenGLNow) {
      stats.gpuMs = gpuBackend.last_gpu_ms();
      stats.gpuTimedSteps = gpuBackend.last_gpu_steps();
    } else if (useCudaNow) {
      stats.gpuMs = cudaBackend.last_gpu_ms();
      stats.gpuTimedSteps = cudaBackend.last_gpu_steps();
    } else {
      stats.gpuMs = -1.0;
      stats.gpuTimedSteps = 0;
    }

    glViewport(0, 0, framebufferWidth, framebufferHeight);
    glClear(GL_COLOR_BUFFER_BIT);

    const float framebufferScaleY =
        windowHeight > 0 ? static_cast<float>(framebufferHeight) /
                               static_cast<float>(windowHeight)
                         : 1.0f;

    if (useOpenGLNow) {
      renderer.render_gpu(gpuBackend.particle_buffer(),
                          gpuBackend.type_buffer(), simulation.particles(),
                          simulation.config(), camera, viewport,
                          framebufferScaleY);
    } else if (useCudaNow) {
      renderer.render_gpu(cudaBackend.particle_buffer(),
                          cudaBackend.type_buffer(), simulation.particles(),
                          simulation.config(), camera, viewport,
                          framebufferScaleY);
    } else {
      renderer.render(simulation.particles(), simulation.config(), camera,
                      viewport, framebufferScaleY);
    }

    draw_world_overlay(simulation.config(), camera, viewport);

    ImGui::PopFont();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);

    const DisplayConfig &display = simulation.config().display;
    if (!display.vsync && display.fpsLimit > 0) {
      const double target = 1.0 / static_cast<double>(display.fpsLimit);
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - frameStart)
                                 .count();
      if (elapsed < target)
        std::this_thread::sleep_for(
            std::chrono::duration<double>(target - elapsed));
    }
  }

  cudaBackend.destroy();
  gpuBackend.destroy();
  renderer.destroy();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}

#ifdef _WIN32
#ifdef PARTICLELIFE_WINDOWS_GUI

#include <Windows.h>

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
  return particlelife_main();
}

#else

int main() { return particlelife_main(); }

#endif
#else

int main() { return particlelife_main(); }

#endif
