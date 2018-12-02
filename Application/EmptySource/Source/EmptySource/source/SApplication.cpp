#include "..\include\SCore.h"
#include "..\include\SMath.h"

#include "..\include\SFileManager.h"

#include "..\include\SWindow.h"
#include "..\include\SApplication.h"
#include "..\include\SMesh.h"
#include "..\include\SShader.h"

SApplication::SApplication() {
	MainWindow = NULL;
	bInitialized = false;
}

bool SApplication::InitalizeGLAD() {
	if (!gladLoadGL()) {
		wprintf(L"Error :: Unable to load OpenGL functions!\n");
		return false;
	}

	glEnable(GL_DEBUG_OUTPUT);
	// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(OGLError, nullptr);
	// Enable all messages, all sources, all levels, and all IDs:
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

	return true;
}

bool SApplication::InitializeWindow() {
	MainWindow = new SWindow();

	if (MainWindow->Create("EmptySource - Debug", ES_WINDOW_MODE_WINDOWED, 1366, 768)) {
		wprintf(L"Error :: Application Window couldn't be created!\n");
		glfwTerminate();
		return false;
	}

	MainWindow->MakeContext();
	MainWindow->InitializeInputs();

	return true;
}

void SApplication::PrintGraphicsInformation() {
	const GLubyte    *renderer = glGetString(GL_RENDERER);
	const GLubyte      *vendor = glGetString(GL_VENDOR);
	const GLubyte     *version = glGetString(GL_VERSION);
	const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	wprintf(L"GC Vendor            : %s\n", FChar((const char*)vendor));
	wprintf(L"GC Renderer          : %s\n", FChar((const char*)vendor));
	wprintf(L"GL Version (string)  : %s\n", FChar((const char*)version));
	wprintf(L"GL Version (integer) : %d.%d\n", major, minor);
	wprintf(L"GLSL Version         : %s\n", FChar((const char*)glslVersion));
}

bool SApplication::InitializeGLFW(unsigned int VersionMajor, unsigned int VersionMinor) {
	if (!glfwInit()) {
		wprintf(L"Error :: Failed to initialize GLFW\n");
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, VersionMajor);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, VersionMinor);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

	glfwSetErrorCallback(&SApplication::GLFWError);
	return true;
}

void SApplication::Initalize() {
	if (bInitialized) return;
	if (InitializeGLFW(4, 6) == false) return; 
	if (InitializeWindow() == false) return;
	if (InitalizeGLAD() == false) return;

	bInitialized = true;
}

void SApplication::Close() {
	MainWindow->Terminate();
	glfwTerminate();
};

void SApplication::MainLoop() {
	///// Temporal Section DELETE AFTER //////

	/*
	* If the first vertex is (-1, -1, 0). This means that unless we transform it in some way,
	* it will be displayed at (-1, -1) on the screen. What does this mean? The screen origin is in the middle,
	* X is on the right, as usual, and Y is up. 
	*/
	SMesh TemporalMesh = SMesh::BuildCube();

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// Perpective matrix (ProjectionMatrix)
	FMatrix4x4 ProjectionMatrix;

	FVector3 EyePosition = FVector3(2, 4, 4);

	// Camera rotation, position Matrix
	FMatrix4x4 ViewMatrix = FMatrix4x4::LookAt(
		EyePosition,        // Camera position
		FVector3(0, 0, 0),	// Look position
		FVector3(0, 1, 0)	// Up vector
	);

	//* Create and compile our GLSL shader program from text files
	// Create the shader
	SShader UnlitBaseShader = SShader(L"Data\\Shaders\\UnlitBase");

	SArray<FMatrix4x4> Matrices;
	Matrices.push_back(FMatrix4x4());

	///////// Create Matrices Buffer //////////////
	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	srand((unsigned int)glfwGetTime());

	///////// Give Uniforms to GLSL /////////////
	// Get the ID of the uniforms
	GLuint    ProjectionMatrixID = UnlitBaseShader.GetLocationID("_ProjectionMatrix");
	GLuint          ViewMatrixID = UnlitBaseShader.GetLocationID("_ViewMatrix");

	//////////////////////////////////////////


	// Activate Z-buffer
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if its closer to the camera
	glDepthFunc(GL_LESS);
	// Draw Mode
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		UnlitBaseShader.Use();

		//////// Drawing ModelMatrix ////////
		ProjectionMatrix = FMatrix4x4::Perspective(
			45.0F * 0.015708F,			// Aperute angle
			MainWindow->AspectRatio(),	// Aspect ratio
			0.1F,						// Near plane
			200.0F						// Far plane
		);

		FVector2 CursorPosition = MainWindow->GetMousePosition();
		EyePosition = FVector3(sinf(float(CursorPosition.x) * 0.01F) * 2, cosf(float(CursorPosition.y) * 0.01F) * 4, cosf(float(CursorPosition.y) * 0.01F) * 4);

		// Camera rotation, position Matrix
		ViewMatrix = FMatrix4x4::LookAt(
			EyePosition,        // Camera position
			FVector3(0, 0, 0),	// Look position
			FVector3(0, 1, 0)	// Up vector
		);

		glUniformMatrix4fv( ProjectionMatrixID, 1, GL_FALSE, ProjectionMatrix.PointerToValue() );
		glUniformMatrix4fv(       ViewMatrixID, 1, GL_FALSE,       ViewMatrix.PointerToValue() );

		for (size_t i = 0; i < 20; i++) {
			Matrices.push_back(
				FMatrix4x4::Translate({ ((rand() % 2000) * 0.5F) - 500, ((rand() % 2000) * 0.5F) - 500, ((rand() % 2000) * 0.5F) - 500 })
			);
		}

		TemporalMesh.BindVertexArray();

		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
		glBufferData(GL_ARRAY_BUFFER, Matrices.size() * sizeof(FMatrix4x4), &Matrices[0], GL_STATIC_DRAW);

		// set transformation matrices as an instance vertex attribute (with divisor 1)
		// note: we're cheating a little by taking the, now publicly declared, VAO of the model's mesh(es) and adding new vertexAttribPointers
		// normally you'd want to do this in a more organized fashion, but for learning purposes this will do.
		// -----------------------------------------------------------------------------------------------------------------------------------
		// set attribute pointers for matrix (4 times vec4)
		glEnableVertexAttribArray(6);
		glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(FMatrix4x4), (void*)0);
		glEnableVertexAttribArray(7);
		glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(FMatrix4x4), (void*)(sizeof(FVector4)));
		glEnableVertexAttribArray(8);
		glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, sizeof(FMatrix4x4), (void*)(2 * sizeof(FVector4)));
		glEnableVertexAttribArray(9);
		glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, sizeof(FMatrix4x4), (void*)(3 * sizeof(FVector4)));

		glVertexAttribDivisor(6, 1);
		glVertexAttribDivisor(7, 1);
		glVertexAttribDivisor(8, 1);
		glVertexAttribDivisor(9, 1);

		// Draw the triangles !
		TemporalMesh.DrawInstanciated((GLsizei)Matrices.size());
		// TemporalMesh.DrawElement();

		glBindVertexArray(0);

		if (MainWindow->GetKeyDown(GLFW_KEY_I)) 
			wprintf(L"Frame Count (%i), Mesh Instances (%i)\n", MainWindow->GetFrameCount(), (int)Matrices.size());

		MainWindow->EndOfFrame();

	} while (
		MainWindow->ShouldClose() == false && !MainWindow->GetKeyDown(GLFW_KEY_ESCAPE)
	);
}

void SApplication::GLFWError(int ErrorID, const char* Description) {
	wprintf(L"Error:: %s", FChar(Description));
}

void APIENTRY SApplication::OGLError(GLenum ErrorSource, GLenum ErrorType, GLuint ErrorID, GLenum ErrorSeverity, GLsizei ErrorLength, const GLchar * ErrorMessage, const void * UserParam)
{
	// ignore non-significant error/warning codes
	if (ErrorID == 131169 || ErrorID == 131185 || ErrorID == 131218 || ErrorID == 131204) return;

	std::wcout << L"Error:: ";

	switch (ErrorSource)
	{
	case GL_DEBUG_SOURCE_API:             std::wcout << L"API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::wcout << L"SYSTEM"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::wcout << L"SHADER_COMPILER"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::wcout << L"THIRD_PARTY"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::wcout << L"APPLICATION"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::wcout << L"OTHER"; break;
	} std::wcout << L"|";

	switch (ErrorType)
	{
	case GL_DEBUG_TYPE_ERROR:               std::wcout << L"ERROR"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::wcout << L"DEPRECATED"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::wcout << L"UNDEFINED"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::wcout << L"PORTABILITY"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::wcout << L"PERFORMANCE"; break;
	case GL_DEBUG_TYPE_MARKER:              std::wcout << L"MARKER"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::wcout << L"PUSH_GROUP"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::wcout << L"POP_GROUP"; break;
	case GL_DEBUG_TYPE_OTHER:               std::wcout << L"OTHER"; break;
	}
	std::wcout << L"(" << ErrorID << L")\n└> " << ErrorMessage << std::endl;
}
