set_project("VParticleLife")
set_version("0.1.0")

set_languages("c++20")

-- MSVC runtime selection is Windows-only.
if is_plat("windows") then
	set_runtimes("MT")
end

add_rules("plugin.compile_commands.autoupdate", {
	outputdir = ".",
})

-- ============================================================
-- Options
-- ============================================================

option("cuda_backend")
set_default(true)
set_showmenu(true)
set_description("Build NVIDIA CUDA compute backend")
option_end()

-- ============================================================
-- Dependencies
-- ============================================================

add_requires("glfw", {
	configs = {
		shared = false,
	},
})

add_requires("glad", {
	configs = {
		api = "gl=4.6",
		profile = "core",
		shared = false,
	},
})

add_requires("imgui", {
	configs = {
		glfw = true,
		opengl3 = true,
		shared = false,
	},
})

-- Cross-platform OpenMP support.
add_requires("openmp")

-- ============================================================
-- Debug
--
-- Unoptimized development build with debug symbols.
-- Shaders and fonts remain external files.
-- ============================================================

target("vpl_debug")
set_kind("binary")
set_basename("VParticleLife-debug")

set_default(false)

set_targetdir("build/bin/debug")
set_rundir("$(projectdir)")

set_symbols("debug")
set_optimize("none")

add_files("src/**.cpp")
add_includedirs("src")

add_packages("glfw", "glad", "imgui", "openmp")

if is_plat("windows") then
	add_syslinks("dxgi")

	add_cxxflags("/Zc:preprocessor", {
		tools = "cl",
		force = true,
	})
end

if has_config("cuda_backend") then
	add_files("src/cuda_compute.cu")
	add_defines("PARTICLELIFE_ENABLE_CUDA")

	-- Target the current GPU only.
	add_cugencodes("native")

	add_cuflags("-G", "-lineinfo", {
		force = true,
	})

	if is_plat("windows") then
		add_cuflags("-Xcompiler=/Zc:preprocessor", {
			force = true,
		})
	end
end

-- ============================================================
-- Native
--
-- Optimized build for the current machine.
-- External resources are kept for active development.
-- ============================================================

target("vpl_native")
set_kind("binary")
set_basename("VParticleLife")

set_default(true)

set_targetdir("build/bin/native")
set_rundir("$(projectdir)")

set_optimize("aggressive")
set_strip("all")

set_policy("build.optimization.lto", true)

add_files("src/**.cpp")
add_includedirs("src")

add_packages("glfw", "glad", "imgui", "openmp")

if is_plat("windows") then
	add_syslinks("dxgi")

	add_cxxflags("/fp:fast", "/arch:AVX512", "/vlen=512", "/Zc:preprocessor", {
		tools = "cl",
		force = true,
	})
elseif is_plat("linux") then
	add_cxxflags("-march=native", "-ffast-math", "-mprefer-vector-width=512", {
		tools = "gcc",
		force = true,
	})

	add_cxxflags("-march=native", "-ffast-math", "-mprefer-vector-width=512", {
		tools = "clang",
		force = true,
	})
end

if has_config("cuda_backend") then
	add_files("src/cuda_compute.cu")
	add_defines("PARTICLELIFE_ENABLE_CUDA")

	-- Target the current GPU only.
	add_cugencodes("native")

	add_cuflags("-O3", "--use_fast_math", {
		force = true,
	})

	if is_plat("windows") then
		add_cuflags("-Xcompiler=/Zc:preprocessor", {
			force = true,
		})
	end
end

-- ============================================================
-- Release
--
-- Distribution build with embedded resources.
-- CPU code stays generic; CUDA includes multiple architectures.
-- ============================================================

target("vpl_release")
set_kind("binary")
set_basename("VParticleLife")

set_default(false)

set_targetdir("dist")
set_rundir("$(projectdir)")

set_optimize("aggressive")
set_strip("all")

set_policy("build.optimization.lto", true)

add_files("src/**.cpp")
add_includedirs("src")

add_packages("glfw", "glad", "imgui", "openmp")

add_defines("PARTICLELIFE_EMBED_RESOURCES")

if is_plat("windows") then
	add_syslinks("dxgi")

	-- Build as a GUI application without a console window.
	add_defines("PARTICLELIFE_WINDOWS_GUI")

	add_cxxflags("/fp:fast", "/Zc:preprocessor", {
		tools = "cl",
		force = true,
	})

	add_ldflags("/SUBSYSTEM:WINDOWS", {
		force = true,
	})
elseif is_plat("linux") then
	-- Keep the release build generic; do not use -march=native.
	add_cxxflags("-ffast-math", {
		tools = "gcc",
		force = true,
	})

	add_cxxflags("-ffast-math", {
		tools = "clang",
		force = true,
	})
end

-- Embed shaders and fonts into the release executable.
add_rules("utils.bin2c", {
	extensions = {
		".vert",
		".frag",
		".comp",
		".ttf",
		-- ".otf",
	},
})

add_files(
	"resources/shaders/**.vert",
	"resources/shaders/**.frag",
	"resources/shaders/**.comp",
	"resources/fonts/**.ttf"
	-- "resources/fonts/**.otf"
)

if has_config("cuda_backend") then
	add_files("src/cuda_compute.cu")
	add_defines("PARTICLELIFE_ENABLE_CUDA")

	-- NVIDIA consumer GPU targets:
	--   sm_75  : Turing
	--   sm_86  : Ampere
	--   sm_89  : Ada
	--   sm_120 : Blackwell
	--   compute_75 provides PTX fallback for newer architectures.
	add_cugencodes("sm_75", "sm_86", "sm_89", "sm_120", "compute_75")

	add_cuflags("-O3", "--use_fast_math", {
		force = true,
	})

	if is_plat("windows") then
		add_cuflags("-Xcompiler=/Zc:preprocessor", {
			force = true,
		})
	end
end
