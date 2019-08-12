#pragma once

#include "Engine/Window.h"

namespace EmptySource {

	class Application {

	public:

		virtual ~Application() = default;

		static Application * GetInstance();

		class RenderPipeline & GetRenderPipeline();

		void OnWindowEvent(WindowEvent& WinEvent);

		void OnInputEvent(InputEvent& InEvent);

		//* Appication Instance
		inline Window & GetWindow() { return *WindowInstance; };

		//* Call this to close the application
		inline void ShouldClose() { bRunning = false; };

		//* The application is running
		bool IsRunning() { return bRunning; }

		//* Entry point of the application
		void Run();

	protected:

		std::unique_ptr<Window> WindowInstance;

		Application();

		virtual void OnInitialize() {};

		virtual void OnAwake() {};

		virtual void OnUpdate() {};

		virtual void OnRender() {};

		virtual void OnTerminate() {};

	private:
		bool bInitialized;

		bool bRunning;

		double RenderTimeSum;

		//* Initialize the application, it creates a window, a context and loads the OpenGL functions.
		void Initalize();

		//* Application loading point
		void Awake();

		//* Application loop
		void UpdateLoop();

		//* Terminates Application
		void Terminate();

		void OnWindowClose(WindowCloseEvent & CloseEvent);
	};

	Application * CreateApplication();

}