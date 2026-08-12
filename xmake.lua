set_project("ParticleLife")
set_version("0.1.0")

set_languages("c++20")

add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", { outputdir = "." })

add_requires("glfw")
add_requires("glad", {
	configs = {
		api = "gl=3.3",
		profile = "core",
	},
})

target("vparticlelife")
set_kind("binary")
add_files("src/**.cpp")
add_includedirs("src")
add_packages("glfw", "glad")
add_cxxflags("/openmp")
set_rundir("$(projectdir)")

if is_mode("release") then
	set_optimize("fastest")
end
