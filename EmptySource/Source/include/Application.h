#pragma once


	class Application {
	private:
		bool bInitialized;
		bool isRunning;
		double RenderTimeSum;

		//* Initialize SDL Functions using OpenGL Versions, returns true if initialized correctly
		bool InitializeSDL(unsigned int VersionMajor, unsigned int VersionMinor);

		//* Initialize GL Functions
		bool InitalizeGL();

		//* Creates the main window for rendering
		bool InitializeWindow();

	protected:

		Application();

		virtual ~Application() = default;

	public:
		static Application & GetInstance();

		class RenderPipeline & GetRenderPipeline();

		struct ContextWindow & GetMainWindow();

		//* Entry point of the application
		void Run();

		//* Initialize the application, it creates a window, a context and loads the OpenGL functions.
		void Initalize();

		//* Application loading point
		void Awake();

		//* Application loop
		void MainLoop();

		//* Terminates Application
		void Terminate();
	};
