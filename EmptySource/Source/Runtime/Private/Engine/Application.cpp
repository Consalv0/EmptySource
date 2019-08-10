
#include "CoreMinimal.h"
#include "Engine/Application.h"
#include "Engine/CoreTime.h"
#include "Engine/Window.h"
#include "Engine/Space.h"
#include "Engine/GameObject.h"
#include "Engine/Transform.h"
#include "Engine/Input.h"

#include "Math/CoreMath.h"
#include "Math/Physics.h"

#include "Utility/TextFormattingMath.h"
#include "Utility/Timer.h"
#if defined(ES_PLATFORM_WINDOWS) & defined(ES_USE_CUDA)
#include "CUDA/CoreCUDA.h"
#endif
#include "Utility/DeviceFunctions.h"

#include "Mesh/Mesh.h"
#include "Mesh/MeshPrimitives.h"

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Material.h"
#include "Rendering/Texture2D.h"
#include "Rendering/Texture3D.h"
#include "Rendering/Cubemap.h"

#include "Files/FileManager.h"

#define RESOURCES_ADD_SHADERSTAGE
#define RESOURCES_ADD_SHADERPROGRAM
#include "Resources/Resources.h"
#include "Resources/MeshLoader.h"
#include "Resources/ImageLoader.h"
#include "Resources/ShaderStageManager.h"

#include "Fonts/Font.h"

#include <SDL.h>

namespace EmptySource {

	Mesh MeshPrimitives::Cube;
	Mesh MeshPrimitives::Quad;

	// int FindBoundingBox(int N, MeshVertex * Vertices);
	// int VoxelizeToTexture3D(Texture3D * Texture, int N, MeshVertex * Vertices);
	// int RTRenderToTexture2D(Texture2D * Texture, std::vector<Vector4> * Spheres, const void * dRandState);
	// const void * GetRandomArray(IntVector2 Dimension);

	Application::Application() {
		bInitialized = false;
		bRunning = false;
		RenderTimeSum = 0;
	}

	void Application::Run() {
		if (bRunning) return;

		bRunning = true;
		LOG_CORE_INFO(L"Initalizing Application:\n");
		Initalize();
		Awake();
		UpdateLoop();
		Terminate();
	}

	Application * Application::GetInstance() {
		static Application * Instance = CreateApplication();
		return Instance;
	}

	RenderPipeline & Application::GetRenderPipeline() {
		static RenderPipeline Pipeline;
		return Pipeline;
	}

	void Application::Initalize() {
		if (bInitialized) return;

		WindowInstance = std::unique_ptr<Window>(Window::Create());
		
		if (!GetWindow().IsRunning()) return;
		if (Debug::InitializeDeviceFunctions() == false) {
			LOG_CORE_WARN(L"Couldn't initialize device functions");
		};

#ifdef ES_USE_CUDA
		CUDA::FindCudaDevice();
#endif

		if (!MeshLoader::Initialize())
			return;

		MeshPrimitives::Initialize();
		Font::InitializeFreeType();

		OnInitialize();

		bInitialized = true;
	}

	void Application::Awake() {
		srand(SDL_GetTicks());
		OnAwake();
	}

	void Application::UpdateLoop() {
		if (!bInitialized) return;

		do {
			MeshLoader::UpdateStatus();
			Time::Tick();

			OnUpdate();

			RenderTimeSum += Time::GetDeltaTime();
			const float MaxFramerate = (1 / 65.F);
			if (RenderTimeSum > MaxFramerate) {
				RenderTimeSum = 0;

				OnRender();

				GetWindow().EndFrame();
				GetRenderPipeline().EndOfFrame();
			}

		} while (
			bRunning == true
		);
	}

	void Application::Terminate() {
		
		ShouldClose();

		Debug::CloseDeviceFunctions();
		MeshLoader::Exit();

		OnTerminate();

		WindowInstance.reset(NULL);

	};

}