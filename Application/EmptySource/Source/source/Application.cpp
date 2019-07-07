#include "../include/Core.h"
#include "../include/Math/CoreMath.h"
#include "../include/Application.h"

#include "../include/Utility/LogGraphics.h"
#include "../include/Utility/DeviceFunctions.h"
#include "../include/Utility/TextFormattingMath.h"
#include "../include/Utility/Timer.h"
#ifndef __APPLE__
#include "../include/CoreCUDA.h"
#endif

#include "../include/CoreTime.h"
#include "../include/Window.h"
#include "../include/FileManager.h"
#include "../include/Mesh.h"
#include "../include/MeshLoader.h"
#include "../include/Utility/MeshPrimitives.h"
#include "../include/Material.h"
#include "../include/ShaderProgram.h"
#include "../include/Space.h"
#include "../include/GameObject.h"
#define RESOURCES_ADD_SHADERSTAGE
#define RESOURCES_ADD_SHADERPROGRAM
#include "../include/Resources.h"
#include "../include/Transform.h"

#include "../include/Font.h"
#include "../include/Text2DGenerator.h"
#include "../include/Graphics.h"
#include "../include/RenderTarget.h"
#include "../include/Texture2D.h"
#include "../include/Texture3D.h"
#include "../include/Cubemap.h"
#include "../include/ImageLoader.h"
#include "../include/Math/Physics.h"
#include "../include/Property.h"

// SDL 2.0.9
#include "../External/SDL/include/SDL.h"
#include "../External/SDL/include/SDL_opengl.h"

#include <thread>
Mesh MeshPrimitives::Cube;
Mesh MeshPrimitives::Quad;

// int FindBoundingBox(int N, MeshVertex * Vertices);
// int VoxelizeToTexture3D(Texture3D * Texture, int N, MeshVertex * Vertices);
// int RTRenderToTexture2D(Texture2D * Texture, std::vector<Vector4> * Spheres, const void * dRandState);
// const void * GetRandomArray(IntVector2 Dimension);

bool CoreApplication::bInitialized = false;
double CoreApplication::RenderTimeSum = 0;

bool CoreApplication::InitalizeGLAD() {
	if (!gladLoadGL()) {
		Debug::Log(Debug::LogCritical, L"Unable to load OpenGL functions!");
		return false;
	}

#ifndef __APPLE__
	glEnable(GL_DEBUG_OUTPUT);
	// glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(Debug::OGLError, nullptr);
	// --- Enable all messages, all sources, all levels, and all IDs:
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#endif

	Debug::PrintGraphicsInformation();

	return true;
}

bool CoreApplication::InitializeSDL(unsigned int VersionMajor, unsigned int VersionMinor) {
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
		Debug::Log(Debug::LogCritical, L"Failed to initialize SDL 2.0.9: %s\n", SDL_GetError());
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, VersionMajor);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, VersionMinor);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

	return true;
}

bool CoreApplication::InitializeWindow() {
	if (!GetMainWindow().Create(
#ifdef _DEBUG
		"EmptySource - Debug",
#else
		"EmptySource",
#endif
		WindowMode::Windowed, 1366, 768))
	{
		return false;
	}

	return true;
}

ContextWindow & CoreApplication::GetMainWindow() {
	static struct ContextWindow MainWindow;
	return MainWindow;
}

void CoreApplication::Initalize() {
	if (bInitialized) return;
#ifdef __APPLE__
    if (InitializeSDL(4, 1) == false) return;
#else
    if (InitializeSDL(4, 6) == false) return;
#endif
	if (InitializeWindow() == false) return;
	if (InitalizeGLAD() == false) return;
    if (Debug::InitializeDeviceFunctions() == false) {
        Debug::Log(Debug::LogWarning, L"Couldn't initialize device functions");
    };

#ifdef WIN32
	CUDA::FindCudaDevice();
#endif

	if (!MeshLoader::Initialize())
		return;
	
	MeshPrimitives::Initialize();
	Font::InitializeFreeType();

	bInitialized = true;
}

void CoreApplication::Terminate() {
    Debug::CloseDeviceFunctions();
	MeshLoader::Exit();
	if (GetMainWindow().IsCreated()) {
		GetMainWindow().Terminate();
	}
	SDL_Quit();
};

#include <iostream>

void CoreApplication::MainLoop() {
	if (!bInitialized) return;

	Property<int> Value(0);

	Observer IntObserver;
	IntObserver.AddCallback([&Value]() { Debug::Log(Debug::LogDebug, L"PropertyInt Changed with value %d", (int)Value); });
	Value.AttachObserver(&IntObserver);

	Value = 1;
	Value = 3;

	Space * OtherNewSpace = Space::CreateSpace(L"MainSpace");
	GGameObject * GameObject = Space::GetMainSpace()->CreateObject<GGameObject>(L"Hola", Transform());
	CComponent * Component = GameObject->CreateComponent<CComponent>();
	Space::Destroy(OtherNewSpace);

	SDL_GL_SetSwapInterval(0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    Debug::Log(Debug::LogDebug, L"%ls", FileManager::GetAppDirectory().c_str());
    
	Font FontFace;
	FontFace.Initialize(FileManager::GetFile(L"Resources/Fonts/SourceSansPro.ttf"));
 
	Text2DGenerator TextGenerator;
	TextGenerator.TextFont = &FontFace;
	TextGenerator.GlyphHeight = 32;
	TextGenerator.AtlasSize = 512;
	TextGenerator.PixelRange = 1.5F;
	TextGenerator.Pivot = 0;

	Bitmap<UCharRGBA> BaseAlbedo;
	Bitmap<UCharRed> BaseMetallic, BaseRoughness;
	Bitmap<UCharRGB> BaseNormal;
	Bitmap<UCharRGB> White, Black;
	Bitmap<FloatRGB> Equirectangular;
	ImageLoader::Load(BaseAlbedo,		FileManager::GetFile(L"Resources/Textures/EscafandraMV1971_BaseColor.png"));
	ImageLoader::Load(BaseMetallic,		FileManager::GetFile(L"Resources/Textures/EscafandraMV1971_Metallic.png"));
	ImageLoader::Load(BaseRoughness,	FileManager::GetFile(L"Resources/Textures/EscafandraMV1971_Roughness.png"));
	ImageLoader::Load(BaseNormal,		FileManager::GetFile(L"Resources/Textures/EscafandraMV1971_Normal.png"));
	ImageLoader::Load(White,			FileManager::GetFile(L"Resources/Textures/White.jpg"));
	ImageLoader::Load(Black,			FileManager::GetFile(L"Resources/Textures/Black.jpg"));
	ImageLoader::Load(Equirectangular,	FileManager::GetFile(L"Resources/Textures/Arches_E_PineTree_3k.hdr"));

	Texture2D EquirectangularTexture = Texture2D(
		IntVector2(Equirectangular.GetWidth(), Equirectangular.GetHeight()),
		Graphics::CF_RGB16F,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_RGB16F,
		Equirectangular.PointerToValue()
	);
	Texture2D BaseAlbedoTexture = Texture2D(
		IntVector2(BaseAlbedo.GetWidth(), BaseAlbedo.GetHeight()),
		Graphics::CF_RGBA,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Border,
		Graphics::CF_RGBA,
		BaseAlbedo.PointerToValue()
	);
	BaseAlbedoTexture.GenerateMipMaps();
	Texture2D BaseMetallicTexture = Texture2D(
		IntVector2(BaseMetallic.GetWidth(), BaseMetallic.GetHeight()),
		Graphics::CF_Red,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_Red,
		BaseMetallic.PointerToValue()
	);
	BaseMetallicTexture.GenerateMipMaps();
	Texture2D BaseRoughnessTexture = Texture2D(
		IntVector2(BaseRoughness.GetWidth(), BaseRoughness.GetHeight()),
		Graphics::CF_Red,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_Red,
		BaseRoughness.PointerToValue()
	);
	BaseRoughnessTexture.GenerateMipMaps();
	Texture2D WhiteTexture = Texture2D(
		IntVector2(White.GetWidth(), White.GetHeight()),
		Graphics::CF_RGB,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_RGB,
		White.PointerToValue()
	);
	WhiteTexture.GenerateMipMaps();
	Texture2D BaseNormalTexture = Texture2D(
		IntVector2(BaseNormal.GetWidth(), BaseNormal.GetHeight()),
		Graphics::CF_RGB,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_RGB,
		BaseNormal.PointerToValue()
	);
	BaseNormalTexture.GenerateMipMaps();
	Texture2D BlackTexture = Texture2D(
		IntVector2(Black.GetWidth(), Black.GetHeight()),
		Graphics::CF_RGB,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Repeat,
		Graphics::CF_RGB,
		Black.PointerToValue()
	);
	BlackTexture.GenerateMipMaps();

	Bitmap<UCharRed> FontAtlas;
	TextGenerator.GenerateGlyphAtlas(FontAtlas);
    
	Texture2D FontMap = Texture2D(
		IntVector2(TextGenerator.AtlasSize),
		Graphics::CF_Red,
		Graphics::FM_MinMagLinear,
		Graphics::AM_Border,
		Graphics::CF_Red,
		FontAtlas.PointerToValue()
	);
	FontMap.GenerateMipMaps();

	/////////// Creating MVP (ModelMatrix, ViewMatrix, Poryection) Matrix //////////////
	// --- Perpective matrix (ProjectionMatrix)
	Matrix4x4 ProjectionMatrix;

	Vector3 EyePosition = { -1.132F , 2.692F , -4.048F };
	Vector3 LightPosition0 = Vector3(2, 1);
	Vector3 LightPosition1 = Vector3(2, 1);
	// --- Camera rotation, position Matrix
	float ViewSpeed = 3;
	Vector3 ViewOrientation;
	Matrix4x4 ViewMatrix;

	// --- Create and compile our GLSL shader programs from text files
	Resource<ShaderProgram> * EquiToCubemapShader = ResourceManager::Load<ShaderProgram>(L"EquirectangularToCubemap");
	Resource<ShaderProgram> * HDRClampingShader   = ResourceManager::Load<ShaderProgram>(L"HDRClampingShader");
	Resource<ShaderProgram> * BRDFShader          = ResourceManager::Load<ShaderProgram>(L"BRDFShader");
	Resource<ShaderProgram> * UnlitShader         = ResourceManager::Load<ShaderProgram>(L"UnLitShader");
	Resource<ShaderProgram> * RenderTextureShader = ResourceManager::Load<ShaderProgram>(L"RenderTextureShader");
	Resource<ShaderProgram> * IntegrateBRDFShader = ResourceManager::Load<ShaderProgram>(L"IntegrateBRDFShader");
	Resource<ShaderProgram> * RenderTextShader    = ResourceManager::Load<ShaderProgram>(L"RenderTextShader");
	Resource<ShaderProgram> * RenderCubemapShader = ResourceManager::Load<ShaderProgram>(L"RenderCubemapShader");

	Material BaseMaterial = Material();
	BaseMaterial.SetShaderProgram(BRDFShader->GetData());
    
	Material UnlitMaterial = Material();
	UnlitMaterial.SetShaderProgram(UnlitShader->GetData());

	Material UnlitMaterialWire = Material();
	UnlitMaterialWire.SetShaderProgram(UnlitShader->GetData());
	UnlitMaterialWire.RenderMode = Graphics::RM_Wire;
	UnlitMaterialWire.CullMode = Graphics::CM_None;

	Material RenderTextureMaterial = Material();
	RenderTextureMaterial.DepthFunction = Graphics::DF_Always;
	RenderTextureMaterial.CullMode = Graphics::CM_None;
	RenderTextureMaterial.SetShaderProgram(RenderTextureShader->GetData());
    
	Material RenderTextMaterial = Material();
	RenderTextMaterial.DepthFunction = Graphics::DF_Always;
	RenderTextMaterial.CullMode = Graphics::CM_None;
	RenderTextMaterial.SetShaderProgram(RenderTextShader->GetData());

	Material RenderCubemapMaterial = Material();
	RenderCubemapMaterial.CullMode = Graphics::CM_None;
	RenderCubemapMaterial.SetShaderProgram(RenderCubemapShader->GetData());

	Material IntegrateBRDFMaterial = Material();
	IntegrateBRDFMaterial.DepthFunction = Graphics::DF_Always;
	IntegrateBRDFMaterial.CullMode = Graphics::CM_None;
	IntegrateBRDFMaterial.SetShaderProgram(IntegrateBRDFShader->GetData());

	Material HDRClampingMaterial = Material();
	HDRClampingMaterial.DepthFunction = Graphics::DF_Always;
	HDRClampingMaterial.CullMode = Graphics::CM_None;
	HDRClampingMaterial.SetShaderProgram(HDRClampingShader->GetData());

	srand(SDL_GetTicks());
	TArray<Mesh> SceneModels;
	TArray<Mesh> LightModels;
	Mesh SphereModel;
	float MeshSelector = 0;

	MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/SphereUV.obj"), true, [&SphereModel](MeshLoader::FileData & ModelData) {
		if (ModelData.bLoaded) {
			SphereModel = Mesh(&(ModelData.Meshes.back()));
			SphereModel.SetUpBuffers();
		}
	});
	MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/Arrow.fbx"), false, [&LightModels](MeshLoader::FileData & ModelData) {
		for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
			LightModels.push_back(Mesh(&(*Data)));
			LightModels.back().SetUpBuffers();
		}
	});
	MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/Sponza.obj"), true, [&SceneModels](MeshLoader::FileData & ModelData) {
		for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
			SceneModels.push_back(Mesh(&(*Data)));
			SceneModels.back().SetUpBuffers();
		}
	});
	MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/EscafandraMV1971.fbx"), true, [&SceneModels](MeshLoader::FileData & ModelData) {
		for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
			SceneModels.push_back(Mesh(&(*Data)));
			SceneModels.back().SetUpBuffers();
		}
	});

	Texture2D RenderedTexture = Texture2D(
	 	IntVector2(GetMainWindow().GetWidth(), GetMainWindow().GetHeight()) / 2, Graphics::CF_RGBA32F, Graphics::FM_MinLinearMagNearest, Graphics::AM_Repeat
	);

	///////// Create Matrices Buffer //////////////
	GLuint ModelMatrixBuffer;
	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);

	Texture2D EquirectangularTextureHDR = Texture2D(
		IntVector2(Equirectangular.GetWidth(), Equirectangular.GetHeight()), Graphics::CF_RGB16F, Graphics::FM_MinMagLinear, Graphics::AM_Repeat
	);
	{
		RenderTarget Renderer = RenderTarget();
		Renderer.SetUpBuffers();
		EquirectangularTextureHDR.Use();
		HDRClampingMaterial.Use();
		HDRClampingMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
		HDRClampingMaterial.SetTexture2D("_EquirectangularMap", &EquirectangularTexture, 0);
		Renderer.Resize(EquirectangularTextureHDR.GetWidth(), EquirectangularTextureHDR.GetHeight());
		MeshPrimitives::Quad.BindVertexArray();
		Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
		HDRClampingMaterial.SetAttribMatrix4x4Array(
			"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
		);

		Renderer.PrepareTexture(&EquirectangularTextureHDR);
		Renderer.Clear();
		MeshPrimitives::Quad.DrawInstanciated(1);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		Renderer.Delete();
		EquirectangularTextureHDR.GenerateMipMaps();
	}

	Texture2D BRDFLut(IntVector2(512), Graphics::CF_RG16F, Graphics::FM_MinMagLinear, Graphics::AM_Clamp);
	{
		RenderTarget Renderer = RenderTarget();
		Renderer.SetUpBuffers();
		Renderer.Use();
		IntegrateBRDFMaterial.Use();
		IntegrateBRDFMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
		Renderer.Resize(BRDFLut.GetWidth(), BRDFLut.GetHeight());
		MeshPrimitives::Quad.BindVertexArray();
		Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
		IntegrateBRDFMaterial.SetAttribMatrix4x4Array(
			"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
		);

		Renderer.PrepareTexture(&BRDFLut);
		Renderer.Clear();
		MeshPrimitives::Quad.DrawInstanciated(1);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		Renderer.Delete();
		// BRDFLut.GenerateMipMaps();
	}

	Cubemap CubemapTexture(Equirectangular.GetHeight() / 2, Graphics::CF_RGB16F, Graphics::FM_MinMagLinear, Graphics::AM_Clamp);
	Cubemap::FromHDREquirectangular(CubemapTexture, &EquirectangularTextureHDR, EquiToCubemapShader->GetData());

	float MaterialMetalness = 1.F;
	float MaterialRoughness = 1.F;
	float LightIntencity = 20.F;

	TArray<Transform> Transforms;
	Transforms.push_back(Transform());

	TArray<int> ElementsIntersected;
	float MultiuseValue = 1;
	const int TextCount = 4;
	float FontSize = 14;
	float FontBoldness = 0.55F;
	WString RenderingText[TextCount];
	Mesh DynamicMesh;
	Point2 TextPivot;

	Transform TestArrowTransform;
	TestArrowTransform.Scale = 0.1F;
	Vector3 TestArrowDirection = 0;
	float TestSphereVelocity = .5F;

	bool bRandomArray = false;
	const void* curandomStateArray = 0;

	glEnable(GL_BLEND);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	do {
		MeshLoader::UpdateStatus();
		Time::Tick();

		ProjectionMatrix = Matrix4x4::Perspective(
			60.0F * MathConstants::DegreeToRad,	// Aperute angle
			GetMainWindow().AspectRatio(),	        // Aspect ratio
			0.03F,						        // Near plane
			1000.0F						        // Far plane
		);

		// --- Camera rotation, position Matrix
		Vector2 CursorPosition = GetMainWindow().GetMousePosition();
		
		Quaternion FrameRotation  = Quaternion::EulerAngles(Vector3(CursorPosition.y, -CursorPosition.x));

		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_W)) {
			Vector3 Forward = FrameRotation * Vector3(0, 0, ViewSpeed);
			EyePosition += Forward * Time::GetDeltaTime() * 
				(!GetMainWindow().GetKeyDown(SDL_SCANCODE_LSHIFT) ? !GetMainWindow().GetKeyDown(SDL_SCANCODE_LCTRL) ? 1.F : .1F : 4.F);
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_A)) {
			Vector3 Right = FrameRotation * Vector3(ViewSpeed, 0, 0);
			EyePosition += Right * Time::GetDeltaTime() *
				(!GetMainWindow().GetKeyDown(SDL_SCANCODE_LSHIFT) ? !GetMainWindow().GetKeyDown(SDL_SCANCODE_LCTRL) ? 1.F : .1F : 4.F);
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_S)) {
			Vector3 Back = FrameRotation * Vector3(0, 0, -ViewSpeed);
			EyePosition += Back * Time::GetDeltaTime() *
				(!GetMainWindow().GetKeyDown(SDL_SCANCODE_LSHIFT) ? !GetMainWindow().GetKeyDown(SDL_SCANCODE_LCTRL) ? 1.F : .1F : 4.F);
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_D)) {
			Vector3 Left = FrameRotation * Vector3(-ViewSpeed, 0, 0);
			EyePosition += Left * Time::GetDeltaTime() *
				(!GetMainWindow().GetKeyDown(SDL_SCANCODE_LSHIFT) ? !GetMainWindow().GetKeyDown(SDL_SCANCODE_LCTRL) ? 1.F : .1F : 4.F);
		}

		Vector3 CameraRayDirection = {
			(2.F * GetMainWindow().GetMousePosition().x) / GetMainWindow().GetWidth() - 1.F,
			1.F - (2.F * GetMainWindow().GetMousePosition().y) / GetMainWindow().GetHeight(),
			-1.F,
		};
		CameraRayDirection = ProjectionMatrix.Inversed() * CameraRayDirection;
		CameraRayDirection.z = -1.F;
		CameraRayDirection = ViewMatrix.Inversed() * CameraRayDirection;
		CameraRayDirection.Normalize();

		ViewMatrix = Matrix4x4::LookAt(EyePosition, EyePosition + FrameRotation * Vector3(0, 0, 1), FrameRotation * Vector3(0, 1));

		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_N)) {
			MaterialMetalness -= 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_M)) {
			MaterialMetalness += 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_E)) {
			MaterialRoughness -= 0.5F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_R)) {
			MaterialRoughness += 0.5F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_L)) {
			LightIntencity += LightIntencity * Time::GetDeltaTime();
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_K)) {
			LightIntencity -= LightIntencity * Time::GetDeltaTime();
		}
		
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_LSHIFT)) {
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_W)) {
				if (BaseMaterial.RenderMode == Graphics::RM_Fill) {
					BaseMaterial.RenderMode = Graphics::RM_Wire;
				} else {
					BaseMaterial.RenderMode = Graphics::RM_Fill;
				}
			}
		}
        
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_UP)) {
            MeshSelector += Time::GetDeltaTime() * 10;
			MeshSelector = MeshSelector > SceneModels.size() - 1 ? SceneModels.size() - 1 : MeshSelector;
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_DOWN)) {
			MeshSelector -= Time::GetDeltaTime() * 10;
			MeshSelector = MeshSelector < 0 ? 0 : MeshSelector;
		}
		if(GetMainWindow().GetKeyDown(SDL_SCANCODE_RIGHT)) {
			MultiuseValue += Time::GetDeltaTime() * MultiuseValue;
		}
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_LEFT)) {
			MultiuseValue -= Time::GetDeltaTime() * MultiuseValue;
		}
		
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_LSHIFT)) {
			if (GetMainWindow().GetKeyDown(SDL_SCANCODE_I)) {
				FontSize += Time::GetDeltaTime() * FontSize;
			}
		
			if (GetMainWindow().GetKeyDown(SDL_SCANCODE_K)) {
				FontSize -= Time::GetDeltaTime() * FontSize;
			}
		} else {
			if (GetMainWindow().GetKeyDown(SDL_SCANCODE_I)) {
				FontBoldness += Time::GetDeltaTime() / 10;
			}
		
			if (GetMainWindow().GetKeyDown(SDL_SCANCODE_K)) {
				FontBoldness -= Time::GetDeltaTime() / 10;
			}
		}
		
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_V)) {
			for (int i = 0; i < 10; i++) {
				RenderingText[1] += (unsigned long)(rand() % 0x3ff);
			}
		}
		
		if (GetMainWindow().GetKeyDown(SDL_SCANCODE_SPACE)) {
			TestArrowTransform.Position = EyePosition;
			TestArrowDirection = CameraRayDirection;
			TestArrowTransform.Rotation = Quaternion::LookRotation(CameraRayDirection, Vector3(0, 1, 0));
		}

		for (int i = 0; i < TextCount; i++) {
			if (TextGenerator.FindCharacters(RenderingText[i]) > 0) {
				TextGenerator.GenerateGlyphAtlas(FontAtlas);
				FontMap.Delete();
				FontMap = Texture2D(
					IntVector2(TextGenerator.AtlasSize),
					Graphics::CF_Red,
					Graphics::FM_MinMagLinear,
					Graphics::AM_Border,
					Graphics::CF_Red,
					FontAtlas.PointerToValue()
				);
				FontMap.GenerateMipMaps();
			}
		}

		// Transforms[0].Rotation = Quaternion::AxisAngle(Vector3(0, 1, 0).Normalized(), Time::GetDeltaTime() * 0.04F) * Transforms[0].Rotation;
		Matrix4x4 TransformMat = Transforms[0].GetLocalToWorldMatrix();
		Matrix4x4 InverseTransform = Transforms[0].GetWorldToLocalMatrix();

		TestArrowTransform.Position += TestArrowDirection * TestSphereVelocity * Time::GetDeltaTime();

		RenderTimeSum += Time::GetDeltaTime();
		const float MaxFramerate = (1 / 65.F);
		if (RenderTimeSum > MaxFramerate) {
			RenderTimeSum = 0;
			size_t TriangleCount = 0;
			size_t VerticesCount = 0;

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glViewport(0, 0, GetMainWindow().GetWidth(), GetMainWindow().GetHeight());
			GetMainWindow().ClearWindow();
		
			Debug::Timer Timer;

			// --- Run the device part of the program
			if (!bRandomArray) {
				// curandomStateArray = GetRandomArray(RenderedTexture.GetDimension());
				bRandomArray = true;
			}
			bool bTestResult = false;

			Ray TestRayArrow(TestArrowTransform.Position, TestArrowDirection);
			if (SceneModels.size() > 100)
				for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)SceneModels.size(); ++MeshCount) {
					BoundingBox3D ModelSpaceAABox = SceneModels[MeshCount].Data.Bounding.Transform(TransformMat);
					TArray<RayHit> Hits;

					if (Physics::RaycastAxisAlignedBox(TestRayArrow, ModelSpaceAABox)) {
						RayHit Hit;
						Ray ModelSpaceCameraRay(
							InverseTransform.MultiplyPoint(TestArrowTransform.Position),
							InverseTransform.MultiplyVector(TestArrowDirection)
						);
						for (MeshFaces::const_iterator Face = SceneModels[MeshCount].Data.Faces.begin(); Face != SceneModels[MeshCount].Data.Faces.end(); ++Face) {
							if (Physics::RaycastTriangle(
								Hit, ModelSpaceCameraRay,
								SceneModels[MeshCount].Data.Vertices[(*Face)[0]].Position,
								SceneModels[MeshCount].Data.Vertices[(*Face)[1]].Position,
								SceneModels[MeshCount].Data.Vertices[(*Face)[2]].Position, BaseMaterial.CullMode != Graphics::CM_Back
							)) {
								Hit.TriangleIndex = int(Face - SceneModels[MeshCount].Data.Faces.begin());
								Hits.push_back(Hit);
							}
						}

						std::sort(Hits.begin(), Hits.end());

						if (Hits.size() > 0 && Hits[0].bHit) {
							Vector3 ArrowClosestContactPoint = TestArrowTransform.Position;
							Vector3 ClosestContactPoint = TestRayArrow.PointAt(Hits[0].Stamp);

							if ((ArrowClosestContactPoint - ClosestContactPoint).MagnitudeSquared() < TestArrowTransform.Scale.x * TestArrowTransform.Scale.x)
							{
								const IntVector3 & Face = SceneModels[MeshCount].Data.Faces[Hits[0].TriangleIndex];
								const Vector3 & N0 = SceneModels[MeshCount].Data.Vertices[Face[0]].Normal;
								const Vector3 & N1 = SceneModels[MeshCount].Data.Vertices[Face[1]].Normal;
								const Vector3 & N2 = SceneModels[MeshCount].Data.Vertices[Face[2]].Normal;
								Vector3 InterpolatedNormal =
									N0 * Hits[0].BaricenterCoordinates[0] +
									N1 * Hits[0].BaricenterCoordinates[1] +
									N2 * Hits[0].BaricenterCoordinates[2];

								Hits[0].Normal = TransformMat.Inversed().Transposed().MultiplyVector(InterpolatedNormal);
								Vector3 ReflectedDirection = Vector3::Reflect(TestArrowDirection, Hits[0].Normal);
								TestArrowDirection = ReflectedDirection.Normalized();
								TestArrowTransform.Rotation = Quaternion::LookRotation(ReflectedDirection, Vector3(0, 1, 0));
							}
						}
					}
				}


			LightPosition0 = Transforms[0].Position + (Transforms[0].Rotation * Vector3(0, 0, 4));
			LightPosition1 = Vector3();

			TArray<Vector4> Spheres = TArray<Vector4>();
			Spheres.push_back(Vector4(Matrix4x4::Translation(Vector3(0, 0, -1.F)) * Vector4(0.F, 0.F, 0.F, 1.F), 0.25F));
			Spheres.push_back(Vector4((Matrix4x4::Translation(Vector3(0, 0, -1.F)) * Quaternion::EulerAngles(
				Vector3(GetMainWindow().GetMousePosition().x, GetMainWindow().GetMousePosition().y, 0)
			).ToMatrix4x4() * Matrix4x4::Translation(Vector3(0.5F))) * Vector4(0.F, 0.F, 0.F, 1.F), 0.5F));
			// bTestResult = RTRenderToTexture2D(&RenderedTexture, &Spheres, curandomStateArray);

			// Framebuffer.Use();

			RenderCubemapMaterial.Use();

			float MaterialRoughnessTemp = (1 - MaterialRoughness) * (CubemapTexture.GetMipmapCount() - 3);
			RenderCubemapMaterial.SetMatrix4x4Array("_ProjectionMatrix", ProjectionMatrix.PointerToValue());
			RenderCubemapMaterial.SetMatrix4x4Array("_ViewMatrix", ViewMatrix.PointerToValue());
			RenderCubemapMaterial.SetTextureCubemap("_Skybox", &CubemapTexture, 0);
			RenderCubemapMaterial.SetFloat1Array("_Roughness", &MaterialRoughnessTemp);

			if (SphereModel.Data.Faces.size() >= 1) {
				SphereModel.SetUpBuffers();
				SphereModel.BindVertexArray();
                
                Matrix4x4 MatrixScale = Matrix4x4::Scaling({ 500, 500, 500 });
				RenderCubemapMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, MatrixScale.PointerToValue(), ModelMatrixBuffer);
				SphereModel.DrawElement();
			}

			BaseMaterial.Use();
			
			BaseMaterial.SetFloat3Array( "_ViewPosition",            EyePosition.PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[0].Position",   LightPosition0.PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[0].Color",    Vector3(0, 0, 1).PointerToValue() );
			BaseMaterial.SetFloat1Array( "_Lights[0].Intencity",                  &LightIntencity );
			BaseMaterial.SetFloat3Array( "_Lights[1].Position",   LightPosition1.PointerToValue() );
			BaseMaterial.SetFloat3Array( "_Lights[1].Color",    Vector3(1, 0, 0).PointerToValue() );
			BaseMaterial.SetFloat1Array( "_Lights[1].Intencity",                  &LightIntencity );
			BaseMaterial.SetFloat1Array( "_Material.Metalness",                &MaterialMetalness );
			BaseMaterial.SetFloat1Array( "_Material.Roughness",                &MaterialRoughness );
			BaseMaterial.SetFloat3Array( "_Material.Color",         Vector3(1.F).PointerToValue() );

			BaseMaterial.SetMatrix4x4Array( "_ProjectionMatrix",       ProjectionMatrix.PointerToValue() );
			BaseMaterial.SetMatrix4x4Array( "_ViewMatrix",                   ViewMatrix.PointerToValue() );
			BaseMaterial.SetTexture2D("_MainTexture", &BaseAlbedoTexture, 0);
			BaseMaterial.SetTexture2D("_NormalTexture", &BaseNormalTexture, 1);
			BaseMaterial.SetTexture2D("_RoughnessTexture", &BaseRoughnessTexture, 2);
			BaseMaterial.SetTexture2D("_MetallicTexture", &BaseMetallicTexture, 3);
			BaseMaterial.SetTexture2D("_BRDFLUT", &BRDFLut, 4);
			BaseMaterial.SetTextureCubemap("_EnviromentMap", &CubemapTexture, 5);
			float CubemapTextureMipmaps = CubemapTexture.GetMipmapCount();
			BaseMaterial.SetFloat1Array("_EnviromentMapLods", &CubemapTextureMipmaps);

			// Transforms[0].Position += Transforms[0].Rotation * Vector3(0, 0, Time::GetDeltaTime() * 2);

			size_t TotalHitCount = 0;
			Ray CameraRay (EyePosition, CameraRayDirection);
			for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)SceneModels.size(); ++MeshCount) {
				const MeshData & ModelData = SceneModels[MeshCount].Data;
				BoundingBox3D ModelSpaceAABox = ModelData.Bounding.Transform(TransformMat);
				TArray<RayHit> Hits;
				
				if (Physics::RaycastAxisAlignedBox(CameraRay, ModelSpaceAABox)) {
					RayHit Hit;
					Ray ModelSpaceCameraRay (
						InverseTransform.MultiplyPoint(EyePosition),
						InverseTransform.MultiplyVector(CameraRayDirection)
					);
					for (MeshFaces::const_iterator Face = ModelData.Faces.begin(); Face != ModelData.Faces.end(); ++Face) {
						if (Physics::RaycastTriangle(
							Hit, ModelSpaceCameraRay,
							ModelData.Vertices[(*Face)[0]].Position,
							ModelData.Vertices[(*Face)[1]].Position,
							ModelData.Vertices[(*Face)[2]].Position, BaseMaterial.CullMode != Graphics::CM_Back
						)) {
							Hit.TriangleIndex = int(Face - ModelData.Faces.begin());
							Hits.push_back(Hit);
						}
					}
					
					std::sort(Hits.begin(), Hits.end()); 
					TotalHitCount += Hits.size();
					
					if (Hits.size() > 0 && Hits[0].bHit) {
						if (LightModels.size() > 0) {
							LightModels[0].SetUpBuffers();
							LightModels[0].BindVertexArray();

							IntVector3 Face = ModelData.Faces[Hits[0].TriangleIndex];
							const Vector3 & N0 = ModelData.Vertices[Face[0]].Normal;
							const Vector3 & N1 = ModelData.Vertices[Face[1]].Normal;
							const Vector3 & N2 = ModelData.Vertices[Face[2]].Normal;
							Vector3 InterpolatedNormal =
								N0 * Hits[0].BaricenterCoordinates[0] +
								N1 * Hits[0].BaricenterCoordinates[1] +
								N2 * Hits[0].BaricenterCoordinates[2];

							Hits[0].Normal = TransformMat.Inversed().Transposed().MultiplyVector(InterpolatedNormal);
							Vector3 ReflectedCameraDir = Vector3::Reflect(CameraRayDirection, Hits[0].Normal);
							Matrix4x4 HitMatrix =
								Matrix4x4::Translation(CameraRay.PointAt(Hits[0].Stamp)) *
								Matrix4x4::Rotation(Quaternion::LookRotation(ReflectedCameraDir, Vector3(0, 1, 0))) *
								Matrix4x4::Scaling(0.07F);
							BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &HitMatrix, ModelMatrixBuffer);

							LightModels[0].DrawInstanciated((GLsizei)1);
							TriangleCount += LightModels[0].Data.Faces.size() * 1;
							VerticesCount += LightModels[0].Data.Vertices.size() * 1;
						}
					}
				}

				SceneModels[MeshCount].SetUpBuffers();
				SceneModels[MeshCount].BindVertexArray();

				BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &TransformMat, ModelMatrixBuffer);
				SceneModels[MeshCount].DrawInstanciated((GLsizei)1);

				TriangleCount += ModelData.Faces.size() * 1;
				VerticesCount += ModelData.Vertices.size() * 1;
			}

			if (LightModels.size() > 0) {
				LightModels[0].SetUpBuffers();
				LightModels[0].BindVertexArray();

				Matrix4x4 ModelMatrix = TestArrowTransform.GetLocalToWorldMatrix();
				BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &ModelMatrix, ModelMatrixBuffer);

				LightModels[0].DrawInstanciated((GLsizei)1);
				TriangleCount += LightModels[0].Data.Faces.size() * 1;
				VerticesCount += LightModels[0].Data.Vertices.size() * 1;
			}

			UnlitMaterialWire.Use();
			
			UnlitMaterialWire.SetMatrix4x4Array("_ProjectionMatrix", ProjectionMatrix.PointerToValue());
			UnlitMaterialWire.SetMatrix4x4Array("_ViewMatrix", ViewMatrix.PointerToValue());
			UnlitMaterialWire.SetFloat3Array("_ViewPosition", EyePosition.PointerToValue());
			
			ElementsIntersected.clear();
			for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)SceneModels.size(); ++MeshCount) {
				BoundingBox3D ModelSpaceAABox = SceneModels[MeshCount].Data.Bounding.Transform(TransformMat);
				if (Physics::RaycastAxisAlignedBox(CameraRay, ModelSpaceAABox)) {
					UnlitMaterialWire.SetFloat4Array("_Material.Color", Vector4(.7F, .2F, .07F, .3F).PointerToValue());
					
					MeshPrimitives::Cube.BindVertexArray();
					ElementsIntersected.push_back(MeshCount);
					Matrix4x4 Transform = Matrix4x4::Translation(ModelSpaceAABox.GetCenter()) * Matrix4x4::Scaling(ModelSpaceAABox.GetSize());
					UnlitMaterialWire.SetAttribMatrix4x4Array("_iModelMatrix", 1, &Transform, ModelMatrixBuffer);
					MeshPrimitives::Cube.DrawInstanciated(1);
				}
			}

			UnlitMaterial.Use();
            
			UnlitMaterial.SetMatrix4x4Array( "_ProjectionMatrix", ProjectionMatrix.PointerToValue() );
			UnlitMaterial.SetMatrix4x4Array( "_ViewMatrix",             ViewMatrix.PointerToValue() );
			UnlitMaterial.SetFloat3Array( "_ViewPosition",             EyePosition.PointerToValue() );
			UnlitMaterial.SetFloat4Array( "_Material.Color", Vector4(LightIntencity, 0, 0, 1).PointerToValue() );
			
			if (LightModels.size() >= 1) {
				LightModels[0].SetUpBuffers();
				LightModels[0].BindVertexArray();

				TArray<Matrix4x4> LightPositions;
				LightPositions.push_back(Matrix4x4::Translation(LightPosition0) * Matrix4x4::Scaling(0.1F));
				LightPositions.push_back(Matrix4x4::Translation(LightPosition1) * Matrix4x4::Scaling(0.1F));

				UnlitMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, &LightPositions[0], ModelMatrixBuffer);

				LightModels[0].DrawInstanciated(2);
			}

			float AppTime = (float)Time::GetEpochTimeMicro() / 1000.F;
			glViewport(0, 0, EquirectangularTextureHDR.GetWidth() / 4 * abs(1 - MultiuseValue), EquirectangularTextureHDR.GetHeight() / 4 * abs(1 - MultiuseValue));
			
			RenderTextureMaterial.Use();

			RenderTextureMaterial.SetFloat1Array("_Time", &AppTime);
			RenderTextureMaterial.SetFloat2Array("_MainTextureSize", EquirectangularTextureHDR.GetDimension().FloatVector2().PointerToValue());
			RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
			RenderTextureMaterial.SetTexture2D("_MainTexture", &EquirectangularTextureHDR, 0);
			float LodLevel = log2f((float)EquirectangularTextureHDR.GetWidth()) * abs(MultiuseValue);
			RenderTextureMaterial.SetFloat1Array("_Lod", &LodLevel);
			
			MeshPrimitives::Quad.BindVertexArray();

			Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
			RenderTextureMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1,
				/*(Quaternion({ MathConstants::HalfPi, 0, 0}).ToMatrix4x4() * */QuadPosition.PointerToValue(),
				ModelMatrixBuffer
			);

			MeshPrimitives::Quad.DrawInstanciated(1);

			glViewport(0, 0, GetMainWindow().GetWidth(), GetMainWindow().GetHeight());
			// --- Activate corresponding render state
			RenderTextMaterial.Use();
			RenderTextMaterial.SetFloat1Array("_Time", &AppTime);
			RenderTextMaterial.SetFloat2Array("_MainTextureSize", FontMap.GetDimension().FloatVector2().PointerToValue());
			RenderTextMaterial.SetMatrix4x4Array("_ProjectionMatrix",
				Matrix4x4::Orthographic(0.F, (float)GetMainWindow().GetWidth(), 0.F, (float)GetMainWindow().GetHeight()).PointerToValue()
			);

			float FontScale = (FontSize / TextGenerator.GlyphHeight);
			RenderTextMaterial.SetTexture2D("_MainTexture", &FontMap, 0);
			RenderTextMaterial.SetFloat1Array("_TextSize", &FontScale);
			RenderTextMaterial.SetFloat1Array("_TextBold", &FontBoldness);

			double TimeCount = 0;
			int TotalCharacterSize = 0;
			DynamicMesh.Clear();
			for (int i = 0; i < TextCount; i++) {
				Timer.Start();
				Vector2 Pivot = TextPivot + Vector2(0.F, GetMainWindow().GetHeight() - (i + 1) * FontSize + FontSize / TextGenerator.GlyphHeight);
				TextGenerator.GenerateMesh( 
					Box2D(0, 0, (float)GetMainWindow().GetWidth(), Pivot.y),
					FontSize, RenderingText[i], &DynamicMesh.Data.Faces, &DynamicMesh.Data.Vertices
				);
				Timer.Stop();
				TimeCount += Timer.GetEnlapsedMili();
				TotalCharacterSize += (int)RenderingText[i].size();
			}
			if (DynamicMesh.SetUpBuffers()) {
				DynamicMesh.BindVertexArray();
				RenderTextMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, Matrix4x4().PointerToValue(), ModelMatrixBuffer);
				DynamicMesh.DrawElement();
			}

			RenderingText[2] = Text::Formatted(
				L"└> Sphere Position(%ls), Sphere2 Position(%ls), ElementsIntersected(%d), RayHits(%d)",
				Text::FormatMath(Spheres[0]).c_str(),
				Text::FormatMath(Spheres[1]).c_str(),
				ElementsIntersected.size(),
				TotalHitCount
			);

			RenderingText[0] = Text::Formatted(
				L"Character(%.2f μs, %d), Temp [%.1f°], %.1f FPS (%.2f ms), Roughness(%.3f), Vertices(%ls), Triangles(%ls), Camera(P%ls, R%ls)",
				TimeCount / double(TotalCharacterSize) * 1000.0,
				TotalCharacterSize,
				Debug::GetDeviceTemperature(0),
				Time::GetFrameRatePerSecond(),
				(1.F / Time::GetFrameRatePerSecond()) * 1000.F,
				MaterialRoughness,
				Text::FormatUnit(VerticesCount, 2).c_str(),
				Text::FormatUnit(TriangleCount, 2).c_str(),
				Text::FormatMath(EyePosition).c_str(),
				Text::FormatMath(Math::ClampAngleComponents(FrameRotation.ToEulerAngles())).c_str()
			);

			GetMainWindow().EndOfFrame();
			glBindVertexArray(0);
		}

		GetMainWindow().PollEvents();


	} while (
		GetMainWindow().ShouldClose() == false && !GetMainWindow().GetKeyDown(SDL_SCANCODE_ESCAPE)
	);
	// delete[] Positions;
}
