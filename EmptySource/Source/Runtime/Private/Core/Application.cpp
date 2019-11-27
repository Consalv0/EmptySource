
#include "CoreMinimal.h"
#include "ImGui/ImGuiLayer.h"
#include "Core/Application.h"
#include "Core/CoreTime.h"
#include "Core/Window.h"
#include "Core/Input.h"

#include "Math/CoreMath.h"

#include "Utility/TextFormattingMath.h"
#if defined(ES_PLATFORM_WINDOWS) & defined(ES_PLATFORM_CUDA)
#include "CUDA/CoreCUDA.h"
#endif

#include "Rendering/Mesh.h"
#include "Rendering/MeshPrimitives.h"

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Shader.h"
#include "Rendering/Material.h"
#include "Rendering/Texture.h"

#include "Files/FileManager.h"

#include "Resources/ModelParser.h"
#include "Resources/ShaderManager.h"

#include "Physics/PhysicsWorld.h"

#include "Fonts/Font.h"

#include <SDL.h>

namespace ESource {

	Mesh MeshPrimitives::Cube;
	Mesh MeshPrimitives::Quad;

	Application::Application() {
		bInitialized = false;
		bRunning = false;
		bRenderImGui = false;
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
		EventDispatcher<WindowEvent> Dispatcher(WinEvent);
		Dispatcher.Dispatch<WindowCloseEvent>(std::bind(&Application::OnWindowClose, this, std::placeholders::_1));
		Dispatcher.Dispatch<WindowResizeEvent>([this](WindowResizeEvent & Event) {
			GetRenderPipeline().bNeedResize = true;
		});

		for (auto LayerIt = AppLayerStack.end(); LayerIt != AppLayerStack.begin(); ) {
			(*--LayerIt)->OnWindowEvent(WinEvent);
		}
	}

	void Application::OnInputEvent(InputEvent & InEvent) {
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
		AudioDeviceInstance = std::make_unique<AudioDevice>();
		DeviceFunctionsInstance = std::unique_ptr<DeviceFunctions>(DeviceFunctions::Create());
		PhysicsWorldInstance = std::make_unique<PhysicsWorld>();

		if (!ModelParser::Initialize())
			return;

		MeshPrimitives::Initialize();
		Font::InitializeFreeType();

		ImGuiLayerInstance = new ImGuiLayer();
		PushLayer(ImGuiLayerInstance);

		OnInitialize();

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

		do {
			Time::Tick();

			ModelParser::UpdateStatus();
			
			for (Layer * LayerIt : AppLayerStack)
				LayerIt->OnUpdate(Time::GetTimeStamp());

			if (!Time::bSkipRender) {
				GetWindow().BeginFrame();
				GetRenderPipeline().BeginFrame();

				for (Layer* LayerPointer : AppLayerStack)
					LayerPointer->OnRender();

				for (Layer* LayerPointer : AppLayerStack)
					LayerPointer->OnPostRender();

				ImGuiLayerInstance->Begin();
				if (bRenderImGui) {
					for (Layer* LayerPointer : AppLayerStack)
						LayerPointer->OnImGuiRender();
				}
				ImGuiLayerInstance->End();

				GetRenderPipeline().EndOfFrame();
				GetWindow().EndFrame();
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

		ModelParser::Exit();

	}

	void Application::OnWindowClose(WindowCloseEvent & CloseEvent) {
		Application::ShouldClose();
	}

}