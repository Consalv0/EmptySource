#pragma once

class SApplication {
public:
	struct SWindow* MainWindow;
	bool bInitialized;

private:
	//* Error Callback related to GLFW
	static void GLFWError(int ErrorID, const char* ErrorDescription);
	
	//* Initialize GLFW Functions using OpenGL Versions, returns true if initialized correctly
	static bool InitializeGLFW(unsigned int VersionMajor, unsigned int VersionMinor);
	
	//* Error Callback related to OpenGL
	static void APIENTRY OGLError(GLenum ErrorSource, GLenum ErrorType, GLuint ErrorID, GLenum ErrorSeverity, GLsizei ErrorLength, const GLchar* ErrorMessage, const void* UserParam);
	
	//* Initialize GLAD OpenGL Functions
	static bool InitalizeGLAD();

	//* Creates the main window for rendering
	bool InitializeWindow();
	
public:

	SApplication();

	//* Prints the Graphics Version info of the videocard used and the GL used
	static void GraphicsInformation();

	//* Initialize the application, it creates a window, a context and loads the OpenGL functions. Returns the error
	void Initalize();
	
	//* Application Loop, draw is done here
	void MainLoop();

	//* Terminates window, 
	void Close();
};