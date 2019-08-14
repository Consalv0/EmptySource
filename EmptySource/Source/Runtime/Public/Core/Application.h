#pragma once

#include "Core/Window.h"
#include "Core/LayerStack.h"

namespace EmptySource {

	class Application {

	public:

		static Application * GetInstance();

		class RenderPipeline & GetRenderPipeline();

		void OnWindowEvent(WindowEvent& WinEvent);

		void OnInputEvent(InputEvent& InEvent);

		void PushLayer(Layer * PushedLayer);

		//* Appication Instance
		inline Window & GetWindow() { return *WindowInstance; };

		//* Application Device Functions
		inline DeviceFunctions & GetDeviceFunctions() { return *DeviceFunctionsInstance; };

		//* Call this to close the application
		inline void ShouldClose() { bRunning = false; };

		//* The application is running
		bool IsRunning() { return bRunning; }

		//* Entry point of the application
		void Run();

		virtual ~Application() = default;

	protected:

		std::unique_ptr<Window> WindowInstance;

		std::unique_ptr<DeviceFunctions> DeviceFunctionsInstance;

		class ImGUILayer * ImGUILayerInstance;

		Application();

	private:
		bool bInitialized;

		bool bRunning;

		LayerStack AppLayerStack;

		//* Initialize the application, it creates a window, a context and loads GL functions.
		void Initalize();

		//* Application loading point, awakens the layers
		void Awake();

		//* Application loop
		void UpdateLoop();

		//* Terminates Application
		void Terminate();

		void OnWindowClose(WindowCloseEvent & CloseEvent);
	};

	Application * CreateApplication();

}