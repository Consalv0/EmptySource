﻿
#include "CoreMinimal.h"
#include "ImGui/ImGuiLayer.h"
#include "Core/Application.h"
#include "Core/CoreTime.h"
#include "Core/Window.h"
#include "Core/Space.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "Math/CoreMath.h"
#include "Math/Physics.h"

#include "Utility/TextFormattingMath.h"
#if defined(ES_PLATFORM_WINDOWS) & defined(ES_PLATFORM_CUDA)
#include "CUDA/CoreCUDA.h"
#endif

#include "Mesh/Mesh.h"
#include "Mesh/MeshPrimitives.h"

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Shader.h"
#include "Rendering/Material.h"
#include "Rendering/Texture2D.h"
#include "Rendering/Texture3D.h"
#include "Rendering/Cubemap.h"

#include "Files/FileManager.h"

#include "Resources/MeshLoader.h"
#include "Resources/ImageLoader.h"
#include "Resources/ShaderManager.h"

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

	void Application::OnWindowEvent(WindowEvent & WinEvent) {
		// LOG_CORE_DEBUG(L"App Window Event : '{}'", WinEvent.GetName());
		EventDispatcher<WindowEvent> Dispatcher(WinEvent);
		Dispatcher.Dispatch<WindowCloseEvent>(std::bind(&Application::OnWindowClose, this, std::placeholders::_1));

		for (auto LayerIt = AppLayerStack.end(); LayerIt != AppLayerStack.begin(); ) {
			(*--LayerIt)->OnWindowEvent(WinEvent);
		}
	}

	void Application::OnInputEvent(InputEvent & InEvent) {
		// LOG_CORE_DEBUG(L"App Input Event : '{}'", InEvent.GetName());
		EventDispatcher<InputEvent> Dispatcher(InEvent);
		Dispatcher.Dispatch<KeyTypedEvent>([](KeyTypedEvent & Event) {
			LOG_CORE_INFO(" Text : '{}'", Event.GetText());
		});

		for (auto LayerIt = AppLayerStack.end(); LayerIt != AppLayerStack.begin(); ) {
			(*--LayerIt)->OnInputEvent(InEvent);
		}
	}

	void Application::PushLayer(Layer * PushedLayer) {
		AppLayerStack.PushLayer(PushedLayer);
		PushedLayer->OnAttach();
		if (bInitialized)
			PushedLayer->OnAwake();
	}

	void Application::Initalize() {
		if (bInitialized) return;

		WindowInstance = std::unique_ptr<Window>(Window::Create());
		WindowInstance->SetWindowEventCallback(std::bind(&Application::OnWindowEvent, this, std::placeholders::_1));
		WindowInstance->SetInputEventCallback(std::bind(&Application::OnInputEvent, this, std::placeholders::_1));
		
		if (!GetWindow().IsRunning()) return;

#ifdef ES_PLATFORM_CUDA
		CUDA::FindCudaDevice();
#endif

		DeviceFunctionsInstance = std::unique_ptr<DeviceFunctions>(DeviceFunctions::Create());

		if (!MeshLoader::Initialize())
			return;

		MeshPrimitives::Initialize();
		Font::InitializeFreeType();

		ImGuiLayerInstance = new ImGuiLayer();
		PushLayer(ImGuiLayerInstance);

		bInitialized = true;
	}

	void Application::Awake() {
		if (!bInitialized) return;

		srand(SDL_GetTicks());

		for (auto LayerIt = AppLayerStack.end(); LayerIt != AppLayerStack.begin(); ) {
			(*--LayerIt)->OnAwake();
		}
	}

	void Application::UpdateLoop() {
		if (!bInitialized) return;

		double RenderTimeSum = 0.0;
		const double MaxFramerate = 1.0 / 45.0;

		do {
			MeshLoader::UpdateStatus();
			Time::Tick();

			for (Layer * LayerIt : AppLayerStack) {
				LayerIt->OnUpdate(Time::GetTimeStamp());
			}

			RenderTimeSum += Time::GetDeltaTime<Time::Second>();
			if (RenderTimeSum > MaxFramerate) {
				RenderTimeSum = 0.0;

				for (Layer* LayerPointer : AppLayerStack)
					LayerPointer->OnRender();

				ImGuiLayerInstance->Begin();
				for (Layer* LayerPointer : AppLayerStack)
					LayerPointer->OnImGuiRender();
				ImGuiLayerInstance->End();

				GetWindow().EndFrame();
				GetRenderPipeline().EndOfFrame();
			}

		} while (
			bRunning == true
		);
	}

	void Application::Terminate() {
		
		ShouldClose();

		for (Layer * LayerIt : AppLayerStack) {
			LayerIt->OnDetach();
		}

		DeviceFunctionsInstance.reset(NULL);
		WindowInstance.reset(NULL);

		AppLayerStack.Clear();

		MeshLoader::Exit();

	}

	void Application::OnWindowClose(WindowCloseEvent & CloseEvent) {
		Application::ShouldClose();
	}

}