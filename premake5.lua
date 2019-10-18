workspace "EmptySource"
	architecture "x64"
    startproject "Sandbox"

	configurations
	{
		"Debug",
		"Release",
		"Shipping"
	}

---%{cfg.system}
outputdir = "%{cfg.buildcfg}_%{cfg.architecture}"

-- Include directories relative to root folder (solution directory)
IncludeDir = {}
IncludeDir["SDL2"] = "EmptySource/External/SDL2/include"
IncludeDir["GLAD"] = "EmptySource/External/GLAD/include"
IncludeDir["FreeType"] = "EmptySource/External/FreeType/include"
IncludeDir["RobinMap"] = "EmptySource/External/RobinMap/include"
IncludeDir["SPDLOG"] = "EmptySource/External/SPDLOG/include"
IncludeDir["STB"] = "EmptySource/External/STB"
IncludeDir["YAML"] = "EmptySource/External/YAML/include"
IncludeDir["Assimp"] = "EmptySource/External/Assimp/include"
IncludeDir["IMGUI"] = "EmptySource/External/IMGUI"
IncludeDir["NVML"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\include"

group "Dependencies"
	--include "EmptySource/External/SDL2/include"
	include "EmptySource/External/GLAD"
	--include "EmptySource/External/STB"
	include "EmptySource/External/IMGUI"
	include "EmptySource/External/YAML"

group ""

project "EmptySource"
	location "EmptySource"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"

	targetdir ("%{prj.name}/Build/" .. outputdir)
	objdir ("%{prj.name}/BinObjs/" .. outputdir)

	pchheader "CoreMinimal.h"
	pchsource "EmptySource/Source/Runtime/Public/CoreMinimal.cpp"

	files {
		"%{prj.name}/Source/**.h",
		"%{prj.name}/Source/**.inl",
		"%{prj.name}/Source/**.cpp",
	}

	includedirs {
        "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.0\\include",
		"%{prj.name}/Source",
		"%{prj.name}/Source/Runtime",
		"%{prj.name}/Source/Runtime/Public",
		"%{IncludeDir.Assimp}",
		"%{IncludeDir.SDL2}",
		"%{IncludeDir.SPDLOG}",
		"%{IncludeDir.GLAD}",
		"%{IncludeDir.FreeType}",
        "%{IncludeDir.RobinMap}",
        "%{IncludeDir.STB}",
		"%{IncludeDir.YAML}",
		"%{IncludeDir.IMGUI}",
		"%{IncludeDir.NVML}",
    }

    libdirs { 
        "%{prj.name}/Libraries"
    }

    links { 
        "SDL2.lib",
        "SDL2main.lib",
        "YAML-CPP",
		"GLAD",
		"IMGUI"
    }

	filter "system:windows"
		systemversion "latest"

		defines {
            "ES_PLATFORM_WINDOWS",
			"ES_DLLEXPORT",
			"ES_PLATFORM_NVML"
		}

	filter "configurations:Debug"
		defines { 
			"ES_DEBUG", "ES_ENABLE_ASSERTS"
		}
		runtime "Debug"
        symbols "on"

	filter "configurations:Release"
		defines "ES_RELEASE"
		runtime "Release"
		optimize "on"

	filter "configurations:Shipping"
		defines "ES_SHIPPING"
		runtime "Release"
		optimize "on"

project "Sandbox"
	location "Sandbox"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"

	targetdir ("%{prj.name}/Build/" .. outputdir)
	objdir ("%{prj.name}/BinObjs/" .. outputdir)

	files {
		"%{prj.name}/Source/**.h",
		"%{prj.name}/Source/**.cpp"
	}

	includedirs {
		"EmptySource/Source",
		"EmptySource/Source/Runtime",
		"EmptySource/Source/Runtime/Public",
        "%{IncludeDir.RobinMap}",
		"%{IncludeDir.SPDLOG}",
		"%{IncludeDir.GLAD}"
	}

    libdirs { 
        "EmptySource/Libraries"
    }

	links {
        "freetype.lib",
		"libfbxsdk-mt.lib",
		"EmptySource"
    }

    configuration "Debug"
        libdirs { 
            "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.0\\lib\\vs2015\\x64\\debug"
		}
		
		links {
			"Assimp\\Debug\\assimp-vc141-mtd.lib"
		}
    
    configuration "Release"
        libdirs {
			"C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.0\\lib\\vs2015\\x64\\release"
		}
		
		links {
			"Assimp\\Release\\assimp-vc141-mt.lib"
		}

    configuration "Shipping"
        libdirs {
            "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.0\\lib\\vs2015\\x64\\release"
        }
		
		links {
			"Assimp\\Release\\assimp-vc141-mt.lib"
		}

	filter "system:windows"
		systemversion "latest"

		defines {
            "ES_PLATFORM_WINDOWS"
		}

		libdirs {
			"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\lib\\x64"
		}

		links {
			"nvml.lib",
		}

	filter "configurations:Debug"
		defines { 
			"ES_DEBUG", "ES_ENABLE_ASSERTS"
		}
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		defines "ES_RELEASE"
		runtime "Release"
		optimize "on"

	filter "configurations:Shipping"
		defines "ES_SHIPPING"
		runtime "Release"
		optimize "on"