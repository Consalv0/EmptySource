workspace "EmptySource"
	architecture "x64"
    startproject "Sandbox"

	configurations
	{
		"Debug",
		"Release",
		"Distrib"
	}

---%{cfg.system}
outputdir = "%{cfg.buildcfg}_%{cfg.architecture}"

-- Include directories relative to root folder (solution directory)
IncludeDir = {}
IncludeDir["SDL2"] = "EmptySource/External/SDL2/include"
IncludeDir["GLAD"] = "EmptySource/External/GLAD/include"
IncludeDir["FreeType"] = "EmptySource/External/FreeType/include"
IncludeDir["RobinMap"] = "EmptySource/External/RobinMap/include/tsl"
IncludeDir["STB"] = "EmptySource/External/STB"
IncludeDir["YAML"] = "EmptySource/External/YAML/include"

group "Dependencies"
	--include "EmptySource/External/SDL2/include"
	include "EmptySource/External/GLAD"
	--include "EmptySource/External/STB"
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

	files {
		"%{prj.name}/Source/**.h",
		"%{prj.name}/Source/**.inl",
		"%{prj.name}/Source/**.cpp",
		"%{prj.name}/External/**/**.h"
	}

	includedirs {
        "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.0\\include",
		"%{prj.name}/Source",
		"%{prj.name}/Source/Runtime",
		"%{prj.name}/Source/Runtime/Public",
        "%{prj.name}/External",
		"%{IncludeDir.SDL2}",
		"%{IncludeDir.GLAD}",
		"%{IncludeDir.FreeType}",
        "%{IncludeDir.RobinMap}",
        "%{IncludeDir.STB}",
        "%{IncludeDir.YAML}"
    }

    libdirs { 
        "%{prj.name}/Libraries"
    }

    links { 
        "SDL2.lib",
        "SDL2main.lib",
        "freetype.lib",
        "libfbxsdk-mt.lib",
        "YAML-CPP",
        "GLAD",
    }

    configuration "Debug"
        libdirs { 
            "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.0\\lib\\vs2015\\x64\\debug"
        }
    
    configuration "Release"
        libdirs {
            "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.0\\lib\\vs2015\\x64\\release"
        }

    configuration "Distrib"
        libdirs {
            "C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.0\\lib\\vs2015\\x64\\release"
        }

	filter "system:windows"
		systemversion "latest"

		defines
		{
            "ES_PLATFORM_WINDOWS",
            "ES_DLLEXPORT"
		}

	filter "configurations:Debug"
		defines "ES_DEBUG"
		runtime "Debug"
        symbols "on"

	filter "configurations:Release"
		defines "ES_RELEASE"
		runtime "Release"
		optimize "on"

	filter "configurations:Distrib"
		defines "ES_DISTRIB"
		runtime "Release"
		optimize "on"

project "Sandbox"
	location "Sandbox"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"
    icon ("%{prj.name}/{prj.name}.ico")

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
		"EmptySource/External",
		"%{IncludeDir.GLAD}"
	}

	links {
		"EmptySource"
    }

	filter "system:windows"
		systemversion "latest"

		defines
		{
            "ES_PLATFORM_WINDOWS"
		}

	filter "configurations:Debug"
		defines "ES_DEBUG"
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		defines "ES_RELEASE"
		runtime "Release"
		optimize "on"

	filter "configurations:Distrib"
		defines "ES_DISTRIB"
		runtime "Release"
		optimize "on"