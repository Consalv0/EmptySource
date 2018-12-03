#include "..\include\Core.h"
#include "..\include\Math\Math.h"

#include "..\include\FileManager.h"

#include "..\include\Window.h"
#include "..\include\Application.h"
#include "..\include\Mesh.h"
#include "..\include\Shader.h"

CoreApplication::CoreApplication() {
	MainWindow = NULL;
	bInitialized = false;
}

bool CoreApplication::InitalizeGLAD() {
	if (!gladLoadGL()) {
		_LOG(LogCritical, L"Unable to load OpenGL functions!");
		return false;
	}

	glEnable(GL_DEBUG_OUTPUT);
	// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(OGLError, nullptr);
	// Enable all messages, all sources, all levels, and all IDs:
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

	return true;
}

bool CoreApplication::InitializeWindow() {
	MainWindow = new ApplicationWindow();

	if (MainWindow->Create("EmptySource - Debug", ES_WINDOW_MODE_WINDOWED, 1366, 768)) {
		_LOG(LogCritical, L"Application Window couldn't be created!");
		glfwTerminate();
		return false;
	}

	MainWindow->MakeContext();
	MainWindow->InitializeInputs();

	return true;
}

void CoreApplication::PrintGraphicsInformation() {
	const GLubyte    *renderer = glGetString(GL_RENDERER);
	const GLubyte      *vendor = glGetString(GL_VENDOR);
	const GLubyte     *version = glGetString(GL_VERSION);
	const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	GLint major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	_LOG(Log, L"GC Vendor            : %s", ToChar((const char*)vendor));
	_LOG(Log, L"GC Renderer          : %s", ToChar((const char*)vendor));
	_LOG(Log, L"GL Version (string)  : %s", ToChar((const char*)version));
	_LOG(Log, L"GL Version (integer) : %d.%d", major, minor);
	_LOG(Log, L"GLSL Version         : %s\n", ToChar((const char*)glslVersion));
}

bool CoreApplication::InitializeGLFW(unsigned int VersionMajor, unsigned int VersionMinor) {
	if (!glfwInit()) {
		_LOG(LogCritical, L"Failed to initialize GLFW\n");
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, VersionMajor);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, VersionMinor);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

	glfwSetErrorCallback(&CoreApplication::GLFWError);
	return true;
}

void CoreApplication::Initalize() {
	if (bInitialized) return;
	if (InitializeGLFW(4, 6) == false) return; 
	if (InitializeWindow() == false) return;
	if (InitalizeGLAD() == false) return;

	bInitialized = true;
}

void CoreApplication::Close() {
	MainWindow->Terminate();
	glfwTerminate();
};

void CoreApplication::MainLoop() {
	///// Temporal Section DELETE LATER //////

	Mesh TemporalMesh = Mesh::BuildCube();

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// Perpective matrix (ProjectionMatrix)
	Matrix4x4 ProjectionMatrix;

	Vector3 EyePosition = Vector3(2, 4, 4);

	// Camera rotation, position Matrix
	Matrix4x4 ViewMatrix;

	//* Create and compile our GLSL shader program from text files
	// Create the shader
	Shader UnlitBaseShader = Shader(L"Data\\Shaders\\UnlitBase");

	TArray<Matrix4x4> Matrices;
	Matrices.push_back(Matrix4x4());

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
		ProjectionMatrix = Matrix4x4::Perspective(
			45.0F * 0.015708F,			// Aperute angle
			MainWindow->AspectRatio(),	// Aspect ratio
			0.1F,						// Near plane
			200.0F						// Far plane
		);

		Vector2 CursorPosition = MainWindow->GetMousePosition();
		EyePosition = Vector3(
			sinf(float(CursorPosition.x) * 0.01F) * 2,
			cosf(float(CursorPosition.y) * 0.01F) * 4,
			cosf(float(CursorPosition.y) * 0.01F) * 4
		);

		// Camera rotation, position Matrix
		ViewMatrix = Matrix4x4::LookAt(
			EyePosition,        // Camera position
			Vector3(0, 0, 0),	// Look position
			Vector3(0, 1, 0)	// Up vector
		);

		glUniformMatrix4fv( ProjectionMatrixID, 1, GL_FALSE, ProjectionMatrix.PointerToValue() );
		glUniformMatrix4fv(       ViewMatrixID, 1, GL_FALSE,       ViewMatrix.PointerToValue() );

		for (size_t i = 0; i < 20; i++) {
			Matrices.push_back(
				Matrix4x4::Translate({ ((rand() % 2000) * 0.5F) - 500, ((rand() % 2000) * 0.5F) - 500, ((rand() % 2000) * 0.5F) - 500 })
			);
		}

		TemporalMesh.BindVertexArray();

		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
		glBufferData(GL_ARRAY_BUFFER, Matrices.size() * sizeof(Matrix4x4), &Matrices[0], GL_STATIC_DRAW);

		// set transformation matrices as an instance vertex attribute (with divisor 1)
		// note: we're cheating a little by taking the, now publicly declared, VAO of the model's mesh(es) and adding new vertexAttribPointers
		// normally you'd want to do this in a more organized fashion, but for learning purposes this will do.
		// -----------------------------------------------------------------------------------------------------------------------------------
		// set attribute pointers for matrix (4 times vec4)
		glEnableVertexAttribArray(6);
		glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)0);
		glEnableVertexAttribArray(7);
		glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(sizeof(Vector4)));
		glEnableVertexAttribArray(8);
		glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(2 * sizeof(Vector4)));
		glEnableVertexAttribArray(9);
		glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(3 * sizeof(Vector4)));

		glVertexAttribDivisor(6, 1);
		glVertexAttribDivisor(7, 1);
		glVertexAttribDivisor(8, 1);
		glVertexAttribDivisor(9, 1);

		// Draw the meshs(es) !
		TemporalMesh.DrawInstanciated((GLsizei)Matrices.size());
		// TemporalMesh.DrawElement();

		glBindVertexArray(0);

		if (MainWindow->GetKeyDown(GLFW_KEY_I)) 
			_LOG(LogDebug, L"Frame Count (%i), Mesh Instances (%i)", MainWindow->GetFrameCount(), (int)Matrices.size());

		MainWindow->EndOfFrame();

	} while (
		MainWindow->ShouldClose() == false && !MainWindow->GetKeyDown(GLFW_KEY_ESCAPE)
	);
}

void CoreApplication::GLFWError(int ErrorID, const char* Description) {
	_LOG(LogError, L"%s", ToChar(Description));
}

void APIENTRY CoreApplication::OGLError(GLenum ErrorSource, GLenum ErrorType, GLuint ErrorID, GLenum ErrorSeverity, GLsizei ErrorLength, const GLchar * ErrorMessage, const void * UserParam)
{
	// ignore non-significant error/warning codes
	if (ErrorID == 131169 || ErrorID == 131185 || ErrorID == 131218 || ErrorID == 131204) return;

	const char* ErrorPrefix = "";

	switch (ErrorType) {
		case GL_DEBUG_TYPE_ERROR:               ErrorPrefix = "error";       break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: ErrorPrefix = "deprecaated";  break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  ErrorPrefix = "undefined";   break;
		case GL_DEBUG_TYPE_PORTABILITY:         ErrorPrefix = "portability"; break;
		case GL_DEBUG_TYPE_PERFORMANCE:         ErrorPrefix = "performance"; break;
		case GL_DEBUG_TYPE_MARKER:              ErrorPrefix = "marker";      break;
		case GL_DEBUG_TYPE_PUSH_GROUP:          ErrorPrefix = "pushgroup";  break;
		case GL_DEBUG_TYPE_POP_GROUP:           ErrorPrefix = "popgroup";   break;
		case GL_DEBUG_TYPE_OTHER:               ErrorPrefix = "other";       break;
	}
	
	_LOG(LogError, L"<%s>(%i) %s", ToChar(ErrorPrefix), ErrorID, ToChar(ErrorMessage));
}
