#pragma once

class CoreApplication {
private:
	static bool bInitialized;
	static double RenderTimeSum;

	static class RenderPipeline * MainRenderPipeline;
	
	//* Initialize SDL Functions using OpenGL Versions, returns true if initialized correctly
	static bool InitializeSDL(unsigned int VersionMajor, unsigned int VersionMinor);
	
	//* Initialize GLAD OpenGL Functions
	static bool InitalizeGLAD();

	//* Creates the main window for rendering
	static bool InitializeWindow();
    
public:
	static RenderPipeline * GetRenderPipeline();

	static void SetRenderPipeline(RenderPipeline * Pipeline);

	static struct ContextWindow & GetMainWindow();

	//* Initialize the application, it creates a window, a context and loads the OpenGL functions.
	static void Initalize();

	//* Application loading point
	static void Awake();
	
	//* Application loop
	static void MainLoop();

	//* Terminates Application
	static void Terminate();
};
