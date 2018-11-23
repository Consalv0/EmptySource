#include "EmptySource/include/EmptyHeaders.h"
#include "..\include\SWindow.h"
#include "..\include\SApplication.h"

SApplication::SApplication() {
	MainWindow = NULL;
}

void SApplication::glfwPrintError(int id, const char* desc) {
	fprintf(stderr, desc);
}

void SApplication::GetGraphicsVersionInformation() {
	const GLubyte    *renderer = glGetString(GL_RENDERER);
	const GLubyte      *vendor = glGetString(GL_VENDOR);
	const GLubyte     *version = glGetString(GL_VERSION);
	const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	printf("GC Vendor            : %s\n", vendor);
	printf("GC Renderer          : %s\n", renderer);
	printf("GL Version (string)  : %s\n", version);
	printf("GL Version (integer) : %d.%d\n", major, minor);
	printf("GLSL Version         : %s\n", glslVersion);
}

int SApplication::Initalize() {
	if (MainWindow != NULL) return 0;
	
	glfwSetErrorCallback(&SApplication::glfwPrintError);
	printf("Initalizing Application:\n");

	MainWindow = new SWindow();

	if (MainWindow->Create("EmptySource - Debug", ES_WINDOW_MODE_WINDOWED, 1080, 720) || MainWindow->Window == NULL) {
		printf("Error :: Application Window couldn't be created!\n");
		printf("\nPress any key to close...\n");
		glfwTerminate();
		_getch();
		return -1;
	}

	MainWindow->MakeContext();
	MainWindow->InitializeInputs();

	if (!gladLoadGL()) {
		printf("Error :: Unable to load OpenGL functions!\n");
		return -1;
	}

	return 1;
}

void SApplication::MainLoop() {
	do {
		glfwSwapBuffers(MainWindow->Window);
		glfwPollEvents();

	} while (
		MainWindow->ShouldClose() == false && 
		glfwGetKey(MainWindow->Window, GLFW_KEY_ESCAPE) != GLFW_PRESS
	);
}

void SApplication::Close() {
	MainWindow->Destroy();
	glfwTerminate();
};