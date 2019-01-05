#include "..\include\Core.h"
#include "..\include\Math\Math.h"

#include "..\include\FileManager.h"
#include "..\include\MeshLoader.h"

#include "..\include\Graphics.h"
#include "..\include\Time.h"
#include "..\include\Window.h"
#include "..\include\Application.h"
#include "..\include\Mesh.h"
#include "..\include\Shader.h"
#include "..\include\Object.h"
#include "..\include\Space.h"
#include "..\include\Material.h"

ApplicationWindow* CoreApplication::MainWindow = NULL;
bool CoreApplication::bInitialized = false;
unsigned long CoreApplication::RenderTimeSum = 0;

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

	if (MainWindow->Create("EmptySource - Debug", WindowMode::Windowed, 1366, 768)) {
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

	_LOG(Log, L"GC Vendor            : %s", CharToWChar((const char*)vendor));
	_LOG(Log, L"GC Renderer          : %s", CharToWChar((const char*)vendor));
	_LOG(Log, L"GL Version (string)  : %s", CharToWChar((const char*)version));
	_LOG(Log, L"GL Version (integer) : %d.%d", major, minor);
	_LOG(Log, L"GLSL Version         : %s\n", CharToWChar((const char*)glslVersion));
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

	Space NewSpace; Space OtherNewSpace(NewSpace);
	Object* GObject = Space::GetFirstSpace()->MakeObject();
	GObject->Delete();
	Mesh TemporalMesh = Mesh::BuildCube();

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// Perpective matrix (ProjectionMatrix)
	Matrix4x4 ProjectionMatrix;

	Vector3 EyePosition = Vector3(2, 4, 4);
	Vector3 LightPosition = Vector3(10, 10);
	// Camera rotation, position Matrix
	Matrix4x4 ViewMatrix;

	//* Create and compile our GLSL shader program from text files
	Shader UnlitBaseShader = Shader(L"Data\\Shaders\\PBRBase");
	UnlitBaseShader.LoadShader(ShaderType::Vertex, L"Data\\Shaders\\PBRBase");
	UnlitBaseShader.LoadShader(ShaderType::Fragment, L"Data\\Shaders\\PBRBase");
	UnlitBaseShader.Compile();
	Material BaseMaterial = Material();
	BaseMaterial.SetShader(&UnlitBaseShader);

	TArray<Matrix4x4> Matrices;
	Matrices.push_back(Matrix4x4());

	///////// Create Matrices Buffer //////////////
	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);

	srand((unsigned int)glfwGetTime());

	///////// Give Uniforms to GLSL /////////////
	// Get the ID of the uniforms
	GLuint    ProjectionMatrixID = UnlitBaseShader.GetLocationID("_ProjectionMatrix");
	GLuint          ViewMatrixID = UnlitBaseShader.GetLocationID("_ViewMatrix");
	GLuint        ViewPositionID = UnlitBaseShader.GetLocationID("_ViewPosition");
	GLuint     Lights0PositionID = UnlitBaseShader.GetLocationID("_Lights[0].Position");
	GLuint    Lights0IntencityID = UnlitBaseShader.GetLocationID("_Lights[0].Intencity");
	GLuint        Lights0ColorID = UnlitBaseShader.GetLocationID("_Lights[0].Color");
	GLuint     Lights1PositionID = UnlitBaseShader.GetLocationID("_Lights[1].Position");
	GLuint    Lights1IntencityID = UnlitBaseShader.GetLocationID("_Lights[1].Intencity");
	GLuint        Lights1ColorID = UnlitBaseShader.GetLocationID("_Lights[1].Color");
	GLuint   MaterialRoughnessID = UnlitBaseShader.GetLocationID("_Material.Roughness");
	GLuint   MaterialMetalnessID = UnlitBaseShader.GetLocationID("_Material.Metalness");
	GLuint       MaterialColorID = UnlitBaseShader.GetLocationID("_Material.Color");

	//////////////////////////////////////////

	MeshFaces OBJFaces; MeshVertices OBJVertices;
	MeshLoader::FromOBJ(FileManager::Open(L"Data\\Models\\SquidwardHouse.obj"), &OBJFaces, &OBJVertices);
	Mesh OBJMesh = Mesh(OBJFaces, OBJVertices);
	MeshFaces SphereFaces; MeshVertices SphereVertices;
	MeshLoader::FromOBJ(FileManager::Open(L"Data\\Models\\Sphere.obj"), &SphereFaces, &SphereVertices);
	Mesh SphereMesh = Mesh(SphereFaces, SphereVertices);
	MeshFaces OtherFaces; MeshVertices OtherVertices;
	MeshLoader::FromOBJ(FileManager::Open(L"Data\\Models\\Cube.obj"), &OtherFaces, &OtherVertices);
	Mesh OtherMesh = Mesh(OtherFaces, OtherVertices);

	do {
		Time::Tick();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		BaseMaterial.Use();

		//////// Drawing ModelMatrix ////////
		ProjectionMatrix = Matrix4x4::Perspective(
			45.0F * 0.015708F,			// Aperute angle
			MainWindow->AspectRatio(),	// Aspect ratio
			0.1F,						// Near plane
			200.0F						// Far plane
		);

		Vector2 CursorPosition = MainWindow->GetMousePosition();
		EyePosition = Vector3(
			sinf(float(CursorPosition.x) * 0.01F) * 4,
			cosf(float(CursorPosition.y) * 0.01F) * 4,
			sinf(float(CursorPosition.y) * 0.01F) * 4
		);

		// Camera rotation, position Matrix
		ViewMatrix = Matrix4x4::LookAt(
			EyePosition,        // Camera position
			Vector3(0, 0, 0),	// Look position
			Vector3(0, 1, 0)	// Up vector
		);

		float MaterialMetalness = 0.98F;
		float MaterialRoughness = 0.01F;
		float LightIntencity = 2000.F;

		glUniformMatrix4fv(  ProjectionMatrixID, 1, GL_FALSE, ProjectionMatrix.PointerToValue() );
		glUniformMatrix4fv(        ViewMatrixID, 1, GL_FALSE,       ViewMatrix.PointerToValue() );
		glUniform3fv(      ViewPositionID, 1,                EyePosition.PointerToValue() );
		glUniform3fv(   Lights0PositionID, 1,              LightPosition.PointerToValue() );
		glUniform3fv(      Lights0ColorID, 1,                 Vector3(1).PointerToValue() );
		glUniform1fv(  Lights0IntencityID, 1,                             &LightIntencity );
		glUniform3fv(   Lights1PositionID, 1,           (-LightPosition).PointerToValue() );
		glUniform3fv(      Lights1ColorID, 1,                 Vector3(1).PointerToValue() );
		glUniform1fv(  Lights1IntencityID, 1,                             &LightIntencity );
		glUniform1fv( MaterialMetalnessID, 1,                          &MaterialMetalness );
		glUniform1fv( MaterialRoughnessID, 1,                          &MaterialRoughness );
		glUniform3fv(     MaterialColorID, 1,     Vector3(0.6F, 0.2F, 0).PointerToValue() );

		OBJMesh.BindVertexArray();

		if (MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT)) {
			if (MainWindow->GetKeyDown(GLFW_KEY_W)) {
				if (BaseMaterial.RenderMode == Render::RenderMode::Fill) {
					BaseMaterial.RenderMode = Render::RenderMode::Wire;
				} else {
					BaseMaterial.RenderMode = Render::RenderMode::Fill;
				}
			}
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_U)) {
			for (size_t i = 0; i < 1; i++) {
				Matrices.push_back(
					Matrix4x4::Translate({ ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128 })
				);
			}
			if (MainWindow->GetKeyDown(GLFW_KEY_Q)) {
				for (size_t i = 0; i < 10000; i++) {
					Matrices.push_back(
						Matrix4x4::Translate({ ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128 })
					);
				}
			}
		}

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
		RenderTimeSum += Time::GetDeltaTimeMilis();
		if (RenderTimeSum > (1000 / 60)) {
			RenderTimeSum = 0;
			OBJMesh.DrawInstanciated((GLsizei)Matrices.size());
			// TemporalMesh.DrawElement();

			MainWindow->SetWindowName(
				TextFormat(L"%s - FPS(%.0f)(%.2f ms), Instances(%d), Triangles(%d), Camera(%s)",
					L"EmptySource",
					Time::GetFrameRate(),
					(1 / Time::GetFrameRate()) * 1000,
					Matrices.size(),
					OBJMesh.Faces.size() * Matrices.size(),
					EyePosition.ToString().c_str()
				)
			);

			MainWindow->EndOfFrame();
		}

		glBindVertexArray(0);

		if (MainWindow->GetKeyDown(GLFW_KEY_W))
			LightPosition += Vector3(100.F, 100.F, 0) * Time::GetDeltaTime();
		if (MainWindow->GetKeyDown(GLFW_KEY_S))
			LightPosition -= Vector3(100.F, 100.F, 0) * Time::GetDeltaTime();
		if (MainWindow->GetKeyDown(GLFW_KEY_A))
			LightPosition += Vector3(0, 0, 100.F) * Time::GetDeltaTime();
		if (MainWindow->GetKeyDown(GLFW_KEY_D))
			LightPosition -= Vector3(0, 0, 100.F) * Time::GetDeltaTime();

		MainWindow->PollEvents();

	} while (
		MainWindow->ShouldClose() == false && !MainWindow->GetKeyDown(GLFW_KEY_ESCAPE)
	);

	UnlitBaseShader.Unload();
}

void CoreApplication::GLFWError(int ErrorID, const char* Description) {
	_LOG(LogError, L"%s", CharToWChar(Description));
}

void APIENTRY CoreApplication::OGLError(GLenum ErrorSource, GLenum ErrorType, GLuint ErrorID, GLenum ErrorSeverity, GLsizei ErrorLength, const GLchar * ErrorMessage, const void * UserParam)
{
	// ignore non-significant error/warning codes
	if (ErrorID == 131169 || ErrorID == 131185 || ErrorID == 131218 || ErrorID == 131204) return;

	const WChar* ErrorPrefix = L"";

	switch (ErrorType) {
		case GL_DEBUG_TYPE_ERROR:               ErrorPrefix = L"error";       break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: ErrorPrefix = L"deprecated";  break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  ErrorPrefix = L"undefined";   break;
		case GL_DEBUG_TYPE_PORTABILITY:         ErrorPrefix = L"portability"; break;
		case GL_DEBUG_TYPE_PERFORMANCE:         ErrorPrefix = L"performance"; break;
		case GL_DEBUG_TYPE_MARKER:              ErrorPrefix = L"marker";      break;
		case GL_DEBUG_TYPE_PUSH_GROUP:          ErrorPrefix = L"pushgroup";  break;
		case GL_DEBUG_TYPE_POP_GROUP:           ErrorPrefix = L"popgroup";   break;
		case GL_DEBUG_TYPE_OTHER:               ErrorPrefix = L"other";       break;
	}
	
	_LOG(LogError, L"<%s>(%i) %s", ErrorPrefix, ErrorID, CharToWChar(ErrorMessage));
}
