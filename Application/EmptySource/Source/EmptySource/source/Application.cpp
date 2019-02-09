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
#include "..\include\ShaderProgram.h"
#include "..\include\Object.h"
#include "..\include\Space.h"
#include "..\include\Material.h"
#include "..\include\Utility\Timer.h"

bool RunTest(int N, MeshVertex * Positions);
int FindBoundingBox(int N, MeshVertex * Vertices);

ApplicationWindow* CoreApplication::MainWindow = NULL;
bool CoreApplication::bInitialized = false;
unsigned long CoreApplication::RenderTimeSum = 0;

bool CoreApplication::InitalizeGLAD() {
	if (!gladLoadGL()) {
		Debug::Log(Debug::LogCritical, L"Unable to load OpenGL functions!");
		return false;
	}

	glEnable(GL_DEBUG_OUTPUT);
	// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(Debug::OGLError, nullptr);
	// Enable all messages, all sources, all levels, and all IDs:
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

	Debug::PrintGraphicsInformation();

	return true;
}

bool CoreApplication::InitializeWindow() {
	MainWindow = new ApplicationWindow();

	if (MainWindow->Create("EmptySource - Debug", WindowMode::Windowed, 1366, 768)) {
		Debug::Log(Debug::LogCritical, L"Application Window couldn't be created!");
		glfwTerminate();
		return false;
	}

	MainWindow->MakeContext();
	MainWindow->InitializeInputs();

	return true;
}

bool CoreApplication::InitializeGLFW(unsigned int VersionMajor, unsigned int VersionMinor) {
	if (!glfwInit()) {
		Debug::Log(Debug::LogCritical, L"Failed to initialize GLFW\n");
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, VersionMajor);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, VersionMinor);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

	glfwSetErrorCallback(&Debug::GLFWError);
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
	
	Space NewSpace; Space OtherNewSpace(NewSpace);
	Object* GObject = Space::GetFirstSpace()->MakeObject();
	GObject->Delete();

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// --- Perpective matrix (ProjectionMatrix)
	Matrix4x4 ProjectionMatrix;

	Vector3 EyePosition = 0;
	Vector3 LightPosition = Vector3(1, 0);
	// --- Camera rotation, position Matrix
	float ViewSpeed = 5;
	Vector3 ViewOrientation;
	Matrix4x4 ViewMatrix;

	// --- Create and compile our GLSL shader program from text files
	Shader VertexBase = Shader(Shader::Type::Vertex, FileManager::Open(L"Data\\Shaders\\Base.vertex.glsl"));
	Shader GeometryVertex = Shader(Shader::Type::Vertex, FileManager::Open(L"Data\\Shaders\\GeometryBase.vertex.glsl"));
	Shader FragmentBRDF = Shader(Shader::Type::Fragment, FileManager::Open(L"Data\\Shaders\\BRDF.fragment.glsl"));
	Shader FragmentUnlit = Shader(Shader::Type::Fragment, FileManager::Open(L"Data\\Shaders\\Unlit.fragment.glsl"));
	Shader Voxelizer = Shader(Shader::Type::Geometry, FileManager::Open(L"Data\\Shaders\\Voxelizer.geometry.glsl"));

	ShaderProgram VoxelBRDFShader = ShaderProgram(L"VoxelBRDF");
	VoxelBRDFShader.Append(&GeometryVertex);
	VoxelBRDFShader.Append(&Voxelizer);
	VoxelBRDFShader.Append(&FragmentBRDF);
	VoxelBRDFShader.Compile();
	
	ShaderProgram BRDFShader = ShaderProgram(L"BRDF");
	BRDFShader.Append(&VertexBase);
	BRDFShader.Append(&FragmentBRDF);
	BRDFShader.Compile();

	ShaderProgram UnlitShader = ShaderProgram(L"UnLit");
	UnlitShader.Append(&VertexBase);
	UnlitShader.Append(&FragmentUnlit);
	UnlitShader.Compile();

	Material BaseMaterial = Material();
	BaseMaterial.SetShaderProgram(&BRDFShader);

	Material VoxelizeMaterial = Material();
	VoxelizeMaterial.SetShaderProgram(&VoxelBRDFShader);

	Material UnlitMaterial = Material();
	UnlitMaterial.SetShaderProgram(&UnlitShader);

	float MaterialMetalness = 0.F;
	float MaterialRoughness = 0.54F;
	float LightIntencity = 100.F;

	TArray<Matrix4x4> Matrices;
	Matrices.push_back(Matrix4x4());

	///////// Create Matrices Buffer //////////////
	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);

	srand((unsigned int)glfwGetTime());

	///////// Get Locations from Shaders /////////////
	// --- Get the ID of the uniforms
	GLuint       ModelMatrixLocation = BRDFShader.GetAttribLocation("_iModelMatrix");
	
	GLuint  ProjectionMatrixLocation = BRDFShader.GetUniformLocation("_ProjectionMatrix");
	GLuint        ViewMatrixLocation = BRDFShader.GetUniformLocation("_ViewMatrix");
	GLuint      ViewPositionLocation = BRDFShader.GetUniformLocation("_ViewPosition");
	GLuint   Lights0PositionLocation = BRDFShader.GetUniformLocation("_Lights[0].Position");
	GLuint  Lights0IntencityLocation = BRDFShader.GetUniformLocation("_Lights[0].Intencity");
	GLuint      Lights0ColorLocation = BRDFShader.GetUniformLocation("_Lights[0].Color");
	GLuint   Lights1PositionLocation = BRDFShader.GetUniformLocation("_Lights[1].Position");
	GLuint  Lights1IntencityLocation = BRDFShader.GetUniformLocation("_Lights[1].Intencity");
	GLuint      Lights1ColorLocation = BRDFShader.GetUniformLocation("_Lights[1].Color");
	GLuint MaterialRoughnessLocation = BRDFShader.GetUniformLocation("_Material.Roughness");
	GLuint MaterialMetalnessLocation = BRDFShader.GetUniformLocation("_Material.Metalness");
	GLuint     MaterialColorLocation = BRDFShader.GetUniformLocation("_Material.Color");

	GLuint LProjectionMatrixLocation = UnlitShader.GetUniformLocation("_ProjectionMatrix");
	GLuint       LViewMatrixLocation = UnlitShader.GetUniformLocation("_ViewMatrix");
	GLuint     LViewPositionLocation = UnlitShader.GetUniformLocation("_ViewPosition");
	GLuint    LMaterialColorLocation = UnlitShader.GetUniformLocation("_Material.Color");

	///////////////////////////////////////////////////

	std::vector<MeshFaces> Faces; std::vector<MeshVertices> Vertices;
	MeshLoader::FromOBJ(FileManager::Open(L"Data\\Models\\Serapis.obj"), &Faces, &Vertices, true);
	std::vector<Mesh> OBJModels;
	float MeshSelector = 0;
	for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
		OBJModels.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]));
	}

	MeshLoader::FromOBJ(FileManager::Open(L"Data\\Models\\Sphere.obj"), &Faces, &Vertices, true);
	std::vector<Mesh> LightModels;
	for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
		LightModels.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount]));
	}

	MeshVertex* Positions = (MeshVertex*)malloc(OBJModels[0].Vertices.size() * sizeof(MeshVertex));
	unsigned long InputTimeSum = 0;

	do {
		Time::Tick();

		//////// Drawing ModelMatrix ////////
		ProjectionMatrix = Matrix4x4::Perspective(
			45.0F * 0.015708F,			// Aperute angle
			MainWindow->AspectRatio(),	// Aspect ratio
			0.1F,						// Near plane
			200.0F						// Far plane
		);

		// --- Camera rotation, position Matrix
		Vector2 CursorPosition = MainWindow->GetMousePosition();
		
		ViewOrientation = { CursorPosition.y * 0.01F, CursorPosition.x * 0.01F, 0.F };
		Quaternion FrameRotation = Quaternion(ViewOrientation.x, { 1, 0, 0 });
		           FrameRotation *= Quaternion(ViewOrientation.y, { 0, 1, 0 });
		
		if (MainWindow->GetKeyDown(GLFW_KEY_W)) {
			Vector3 Forward = FrameRotation.ToMatrix4x4() * Vector3(0, 0, ViewSpeed);
			EyePosition += Forward * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_A)) {
			Vector3 Right = FrameRotation.ToMatrix4x4() * Vector3(ViewSpeed, 0, 0);
			EyePosition += Right * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_S)) {
			Vector3 Back = FrameRotation.ToMatrix4x4() * Vector3(0, 0, -ViewSpeed);
			EyePosition += Back * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_D)) {
			Vector3 Left = FrameRotation.ToMatrix4x4() * Vector3(-ViewSpeed, 0, 0);
			EyePosition += Left * Time::GetDeltaTime();
		}
		ViewMatrix =
			Matrix4x4::Translate(EyePosition) * FrameRotation.ToMatrix4x4();

		if (MainWindow->GetKeyDown(GLFW_KEY_N)) {
			MaterialMetalness -= 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_M)) {
			MaterialMetalness += 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_E)) {
			MaterialRoughness -= 0.1F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_R)) {
			MaterialRoughness += 0.1F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_L)) {
			LightIntencity += 10 * Time::GetDeltaTime();
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_K)) {
			LightIntencity -= 10 * Time::GetDeltaTime();
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_LEFT_SHIFT)) {
			if (MainWindow->GetKeyDown(GLFW_KEY_W)) {
				if (BaseMaterial.RenderMode == Render::RenderMode::Fill) {
					BaseMaterial.RenderMode = Render::RenderMode::Wire;
				} else {
					BaseMaterial.RenderMode = Render::RenderMode::Fill;
				}
			}
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_UP)) {
			MeshSelector += Time::GetDeltaTime() * 10;
			MeshSelector = MeshSelector > OBJModels.size() - 1 ? OBJModels.size() - 1 : MeshSelector;
		}
		if (MainWindow->GetKeyDown(GLFW_KEY_DOWN)) {
			MeshSelector -= Time::GetDeltaTime() * 10;
			MeshSelector = MeshSelector < 0 ? 0 : MeshSelector;
		}

		if (MainWindow->GetKeyDown(GLFW_KEY_U)) {
			InputTimeSum += Time::GetDeltaTimeMilis();
			if (InputTimeSum > (300)) {
				for (size_t i = 0; i < 1; i++) {
					Matrices.push_back(
						Matrix4x4::Translate({ ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128 })
					);
				}
			}
			if (MainWindow->GetKeyDown(GLFW_KEY_Q)) {
				for (size_t i = 0; i < 100; i++) {
					Matrices.push_back(
						Matrix4x4::Translate({ ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128, ((rand() % 500) * 0.5F) - 128 })
					);
				}
			}
		}

		// --- Draw the meshs(es) !
		RenderTimeSum += Time::GetDeltaTimeMilis();
		if (RenderTimeSum > (1000 / 60)) {
			RenderTimeSum = 0;
			size_t TriangleCount = 0;
			size_t VerticesCount = 0;

			MainWindow->ClearWindow();

			Debug::Timer Timer;
			Timer.Start();

			// Debug::Log(Debug::LogDebug, L"Test Host Vector[%d] %s", 0, OBJModels[0].Vertices[0].Position.ToString().c_str());

			std::memcpy(Positions, &OBJModels[0].Vertices[0], OBJModels[0].Vertices.size() * sizeof(MeshVertex));

			// --- Run the device part of the program
			bool bTestResult;
			bTestResult = FindBoundingBox((int)OBJModels[0].Vertices.size(), Positions);

			// Debug::Log(Debug::LogDebug, L"Test Device Vector[%d] %s", 0, Positions[0].Position.ToString().c_str());
			
			Timer.Stop();
			Debug::Log(
				Debug::LogWarning, L"CUDA Test with %s elements durantion: %dms",
				Text::FormattedUnit(OBJModels[0].Vertices.size(), 2).c_str(),
				Timer.GetEnlapsed()
			);
			Debug::Log(Debug::NoLog, L"\n");

			// VoxelizeMaterial.Use();
			// 
			// glUniformMatrix4fv(ProjectionMatrixLocation, 1, GL_FALSE, ProjectionMatrix.PointerToValue());
			// glUniformMatrix4fv(ViewMatrixLocation, 1, GL_FALSE, ViewMatrix.PointerToValue());
			// glUniform3fv(ViewPositionLocation, 1, EyePosition.PointerToValue());
			// glUniform3fv(Lights0PositionLocation, 1, LightPosition.PointerToValue());
			// glUniform3fv(Lights0ColorLocation, 1, Vector3(1).PointerToValue());
			// glUniform1fv(Lights0IntencityLocation, 1, &LightIntencity);
			// glUniform3fv(Lights1PositionLocation, 1, (-LightPosition).PointerToValue());
			// glUniform3fv(Lights1ColorLocation, 1, Vector3(1).PointerToValue());
			// glUniform1fv(Lights1IntencityLocation, 1, &LightIntencity);
			// glUniform1fv(MaterialMetalnessLocation, 1, &MaterialMetalness);
			// glUniform1fv(MaterialRoughnessLocation, 1, &MaterialRoughness);
			// glUniform3fv(MaterialColorLocation, 1, Vector3(0.6F, 0.2F, 0).PointerToValue());
			// 
			// for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)OBJModels.size(); ++MeshCount) {
			// 	OBJModels[MeshCount].BindVertexArray();
			// 
			// 	BRDFShader.SetMatrix4x4Array(ModelMatrixLocation, (int)Matrices.size(), &Matrices[0], ModelMatrixBuffer);
			// 
			// 	OBJModels[MeshCount].DrawInstanciated((GLsizei)Matrices.size());
			// 	TriangleCount += OBJModels[MeshCount].Faces.size() * Matrices.size();
			// }

			BaseMaterial.Use();

			glUniformMatrix4fv(  ProjectionMatrixLocation, 1, GL_FALSE, ProjectionMatrix.PointerToValue() );
			glUniformMatrix4fv(        ViewMatrixLocation, 1, GL_FALSE,       ViewMatrix.PointerToValue() );
			glUniform3fv(      ViewPositionLocation, 1,                EyePosition.PointerToValue() );
			glUniform3fv(   Lights0PositionLocation, 1,              LightPosition.PointerToValue() );
			glUniform3fv(      Lights0ColorLocation, 1,                 Vector3(1).PointerToValue() );
			glUniform1fv(  Lights0IntencityLocation, 1,                             &LightIntencity );
			glUniform3fv(   Lights1PositionLocation, 1,           (-LightPosition).PointerToValue() );
			glUniform3fv(      Lights1ColorLocation, 1,                 Vector3(1).PointerToValue() );
			glUniform1fv(  Lights1IntencityLocation, 1,                             &LightIntencity );
			glUniform1fv( MaterialMetalnessLocation, 1,                          &MaterialMetalness );
			glUniform1fv( MaterialRoughnessLocation, 1,                          &MaterialRoughness );
			glUniform3fv(     MaterialColorLocation, 1,     Vector3(0.6F, 0.2F, 0).PointerToValue() );

			for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)OBJModels.size(); ++MeshCount) {
				OBJModels[MeshCount].BindVertexArray();

				BRDFShader.SetMatrix4x4Array(ModelMatrixLocation, (int)Matrices.size(), &Matrices[0], ModelMatrixBuffer);

				OBJModels[MeshCount].DrawInstanciated((GLsizei)Matrices.size());
				TriangleCount += OBJModels[MeshCount].Faces.size() * Matrices.size();
				VerticesCount += OBJModels[MeshCount].Vertices.size() * Matrices.size();
			}
			
			UnlitMaterial.Use();

			glUniformMatrix4fv( LProjectionMatrixLocation, 1, GL_FALSE, ProjectionMatrix.PointerToValue() );
			glUniformMatrix4fv(       LViewMatrixLocation, 1, GL_FALSE,       ViewMatrix.PointerToValue() );
			glUniform3fv(  LViewPositionLocation, 1, EyePosition.PointerToValue() );
			glUniform3fv( LMaterialColorLocation, 1,  Vector3(1).PointerToValue() );
			
			LightModels[0].BindVertexArray();

			vector<Matrix4x4> LightPositions;
			LightPositions.push_back(Matrix4x4::Scale(0.1F) * Matrix4x4::Translate(LightPosition));
			LightPositions.push_back(Matrix4x4::Scale(0.1F) * Matrix4x4::Translate(-LightPosition));
			BRDFShader.SetMatrix4x4Array(ModelMatrixLocation, 2, &LightPositions[0], ModelMatrixBuffer);

			LightModels[0].DrawInstanciated(2);

			MainWindow->SetWindowName(
				Text::Formatted(L"%s - FPS(%.0f)(%.2f ms), Instances(%s), Vertices(%s), Triangles(%s), Camera(%s, %s)",
					L"EmptySource",
					Time::GetFrameRate(),
					(1 / Time::GetFrameRate()) * 1000,
					Text::FormattedUnit(Matrices.size(), 2).c_str(),
					Text::FormattedUnit(VerticesCount, 2).c_str(),
					Text::FormattedUnit(TriangleCount, 2).c_str(),
					EyePosition.ToString().c_str(),
					Math::ClampAngleComponents(FrameRotation.ToEulerAngles() * MathConstants::RadToDegree).ToString().c_str()
				)
			);

			MainWindow->EndOfFrame();
		}

		glBindVertexArray(0);

		MainWindow->PollEvents();

	} while (
		MainWindow->ShouldClose() == false && !MainWindow->GetKeyDown(GLFW_KEY_ESCAPE)
	);

	delete[] Positions;
	BRDFShader.Unload();
}