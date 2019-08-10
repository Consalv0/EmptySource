
#include "Engine/EmptySource.h"
#include "Engine/Core.h"
#include "Engine/CoreTime.h"
#include "Engine/Window.h"
#include "Engine/Space.h"
#include "Engine/GameObject.h"
#include "Engine/Transform.h"
#include "Engine/Input.h"

#include "Math/CoreMath.h"
#include "Math/Physics.h"

#include "Utility/TextFormattingMath.h"
#include "Utility/Timer.h"
#if defined(_WIN32) & defined(USE_CUDA)
#include "CUDA/CoreCUDA.h"
#endif
#include "Utility/DeviceFunctions.h"

#include "Mesh/Mesh.h"
#include "Mesh/MeshPrimitives.h"

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderTarget.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/ShaderProgram.h"
#include "Rendering/Material.h"
#include "Rendering/Texture2D.h"
#include "Rendering/Texture3D.h"
#include "Rendering/Cubemap.h"

#include "Files/FileManager.h"

#define RESOURCES_ADD_SHADERSTAGE
#define RESOURCES_ADD_SHADERPROGRAM
#include "Resources/Resources.h"
#include "Resources/MeshLoader.h"
#include "Resources/ImageLoader.h"
#include "Resources/ShaderStageManager.h"

#include "Components/ComponentRenderer.h"

#include "Fonts/Font.h"
#include "Fonts/Text2DGenerator.h"

#include "Events/Property.h"

#include "../External/GLAD/include/glad/glad.h"

#include "../External/SDL2/include/SDL_keycode.h"
#include "../External/SDL2/include/SDL_audio.h"

using namespace EmptySource;

class SandboxApplication : public EmptySource::Application {
private:

	Font FontFace;
	Text2DGenerator TextGenerator;
	Bitmap<UCharRed> FontAtlas;
	Bitmap<UCharRGBA> BaseAlbedo, FlamerAlbedo;
	Bitmap<UCharRed> BaseMetallic, BaseRoughness, FlamerMetallic, FlamerRoughness;
	Bitmap<UCharRGB> BaseNormal, FlamerNormal;
	Bitmap<UCharRGB> White, Black;
	Bitmap<FloatRGB> Equirectangular;

	// --- Perpective matrix (ProjectionMatrix)
	GLuint ModelMatrixBuffer;
	size_t TriangleCount = 0;
	size_t VerticesCount = 0;
	Matrix4x4 ProjectionMatrix;

	Vector3 EyePosition = { -1.132F , 2.692F , -4.048F };
	Vector3 LightPosition0 = Vector3(2, 1);
	Vector3 LightPosition1 = Vector3(2, 1);

	TArray<Mesh> SceneModels;
	TArray<Mesh> LightModels;
	Mesh SphereModel;
	float MeshSelector = 0;
	
	// --- Camera rotation, position Matrix
	float ViewSpeed = 3;
	Vector3 ViewOrientation;
	Matrix4x4 ViewMatrix; 
	Quaternion FrameRotation; 
	Vector2 CursorPosition;

	Material UnlitMaterial = Material();
	Material UnlitMaterialWire = Material();
	Material RenderTextureMaterial = Material();
	Material RenderTextMaterial = Material();
	Material RenderCubemapMaterial = Material();
	Material IntegrateBRDFMaterial = Material();
	Material HDRClampingMaterial = Material();
	Material BaseMaterial = Material();

	float MaterialMetalness = 1.F;
	float MaterialRoughness = 1.F;
	float LightIntencity = 20.F;

	TArray<Transform> Transforms; 
	Matrix4x4 TransformMat;
	Matrix4x4 InverseTransform; 
	Vector3 CameraRayDirection;
	
	TArray<int> ElementsIntersected;
	float MultiuseValue = 1;
	static const int TextCount = 4;
	float FontSize = 14;
	float FontBoldness = 0.55F;
	WString RenderingText[TextCount];
	Mesh DynamicMesh;
	Point2 TextPivot;

	Texture2D * EquirectangularTextureHDR;
	Texture2D * FontMap;
	Cubemap * CubemapTexture;

	Transform TestArrowTransform;
	Vector3 TestArrowDirection = 0;
	float TestSphereVelocity = .5F;

	bool bRandomArray = false;
	const void* curandomStateArray = 0;

protected:

	virtual void OnInitialize() override { };

	virtual void OnAwake() override {
		GetRenderPipeline().AddStage(L"TestStage", new RenderStage());

		Space * OtherNewSpace = Space::CreateSpace(L"MainSpace");
		GGameObject * GameObject = Space::GetMainSpace()->CreateObject<GGameObject>(L"SkyBox", Transform());
		CComponent * Component = GameObject->CreateComponent<CRenderer>();

		static Uint32 SampleLength; // length of our sample
		static Uint8 * SampleBuffer; // buffer containing our audio file
		static SDL_AudioSpec SampleSpecs; // the specs of our piece of music
		static Uint8 * AudioPosition; // global pointer to the audio buffer to be played
		static Uint32  AudioLength; // remaining length of the sample we have to play
		FileStream * SampleFile = FileManager::GetFile(L"Resources/Sounds/6503.wav");
		if (SampleFile != NULL) {
			SampleFile->Close();
		}

		if (SampleFile == NULL || SDL_LoadWAV(Text::WideToNarrow(SampleFile->GetPath()).c_str(), &SampleSpecs, &SampleBuffer, &SampleLength) == NULL) {
			LOG_ERROR("Couldn't not open the sound file or is invalid");
		}
		else {
			{
				static SDL_AudioCVT AudioConvert;
				if (SDL_BuildAudioCVT(&AudioConvert, SampleSpecs.format, SampleSpecs.channels, SampleSpecs.freq,
					AUDIO_F32LSB, SampleSpecs.channels, SampleSpecs.freq)) {
					AudioConvert.len = SampleLength;
					AudioConvert.buf = (Uint8 *)SDL_malloc(SampleLength * AudioConvert.len_mult);
					SDL_memcpy(AudioConvert.buf, SampleBuffer, SampleLength);
					SDL_ConvertAudio(&AudioConvert);
					SDL_FreeWAV(SampleBuffer);
					SampleSpecs.format = AUDIO_F32LSB;
					SampleBuffer = AudioConvert.buf;
					SampleLength = AudioConvert.len_cvt;
				}
			}
			// Set the callback function
			SampleSpecs.callback = [](void *UserData, Uint8 *Stream, int Length) -> void {
				/* Silence the main buffer */
				SDL_memset(Stream, 0, Length);

				if (SampleLength <= 0) {
					return;
				}

				Length = ((uint32_t)Length > AudioLength) ? AudioLength : (uint32_t)Length;
				// SDL_memcpy(Stream, AudioPosition, Length); 		    // simply copy from one buffer into the other
				try {
					SDL_MixAudioFormat(Stream, AudioPosition, SampleSpecs.format, Length, SDL_MIX_MAXVOLUME / 4);	// mix from one buffer into another
				}
				catch (...) {
					LOG_DEBUG("Error in Audio Data");
				}

				AudioPosition += Length;
				AudioLength -= Length;

				if (AudioLength <= 0) {
					// SDL_PauseAudio(1);
					AudioPosition = SampleBuffer; // copy sound buffer
					AudioLength = SampleLength; // copy file length
				}
			};

			SampleSpecs.userdata = NULL;
			// Set the variables
			AudioPosition = SampleBuffer; // copy sound buffer
			AudioLength = SampleLength; // copy file length

			/* Open the audio device */
			if (SDL_OpenAudio(&SampleSpecs, NULL) < 0) {
				LOG_ERROR("Couldn't open audio: {0}\n", SDL_GetError());
				exit(-1);
			}
			/* Start playing */
			SDL_PauseAudio(false);

			int SampleSize = SDL_AUDIO_BITSIZE(SampleSpecs.format);
			bool IsFloat = SDL_AUDIO_ISFLOAT(SampleSpecs.format);
			LOG_DEBUG("Audio Time Length: {:0.3f}", (SampleLength * 8 / (SampleSize * SampleSpecs.channels)) / (float)SampleSpecs.freq);
		}

		LOG_DEBUG(L"{0}", FileManager::GetAppDirectory());

		{
			Property<int> Value(0);

			Observer IntObserver;
			IntObserver.AddCallback("Test", [&Value]() { LOG_DEBUG("PropertyInt Changed with value {0:d}", (int)Value); });
			Value.AttachObserver(&IntObserver);

			Value = 1;
			Value = 3;
		}

		FontFace.Initialize(FileManager::GetFile(L"Resources/Fonts/SourceSansPro.ttf"));

		TextGenerator.TextFont = &FontFace;
		TextGenerator.GlyphHeight = 32;
		TextGenerator.AtlasSize = 512;
		TextGenerator.PixelRange = 1.5F;
		TextGenerator.Pivot = 0;

		ImageLoader::Load(BaseAlbedo, FileManager::GetFile(L"Resources/Textures/Sponza/sponza_column_a_diff.png"));
		ImageLoader::Load(BaseMetallic, FileManager::GetFile(L"Resources/Textures/Sponza/sponza_arch_spec.png"));
		ImageLoader::Load(BaseRoughness, FileManager::GetFile(L"Resources/Textures/Sponza/sponza_column_a_roug.jpg"));
		ImageLoader::Load(BaseNormal, FileManager::GetFile(L"Resources/Textures/Sponza/sponza_column_a_normal.png"));
		ImageLoader::Load(FlamerAlbedo, FileManager::GetFile(L"Resources/Textures/EscafandraMV1971_BaseColor.png"));
		ImageLoader::Load(FlamerMetallic, FileManager::GetFile(L"Resources/Textures/EscafandraMV1971_Metallic.png"));
		ImageLoader::Load(FlamerRoughness, FileManager::GetFile(L"Resources/Textures/EscafandraMV1971_Roughness.png"));
		ImageLoader::Load(FlamerNormal, FileManager::GetFile(L"Resources/Textures/EscafandraMV1971_Normal.png"));
		ImageLoader::Load(White, FileManager::GetFile(L"Resources/Textures/White.jpg"));
		ImageLoader::Load(Black, FileManager::GetFile(L"Resources/Textures/Black.jpg"));
		ImageLoader::Load(Equirectangular, FileManager::GetFile(L"Resources/Textures/glacier.hdr"));

		Texture2D EquirectangularTexture = Texture2D(
			IntVector2(Equirectangular.GetWidth(), Equirectangular.GetHeight()),
			CF_RGB16F,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_RGB16F,
			Equirectangular.PointerToValue()
		);
		Texture2D * BaseAlbedoTexture = new Texture2D(
			IntVector2(BaseAlbedo.GetWidth(), BaseAlbedo.GetHeight()),
			CF_RGBA,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_RGBA,
			BaseAlbedo.PointerToValue()
		);
		BaseAlbedoTexture->GenerateMipMaps();
		OldResourceManager::Load(L"BaseAlbedoTexture", BaseAlbedoTexture);
		Texture2D * BaseMetallicTexture = new Texture2D(
			IntVector2(BaseMetallic.GetWidth(), BaseMetallic.GetHeight()),
			CF_Red,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_Red,
			BaseMetallic.PointerToValue()
		);
		BaseMetallicTexture->GenerateMipMaps();
		OldResourceManager::Load(L"BaseMetallicTexture", BaseMetallicTexture);
		Texture2D * BaseRoughnessTexture = new Texture2D(
			IntVector2(BaseRoughness.GetWidth(), BaseRoughness.GetHeight()),
			CF_Red,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_Red,
			BaseRoughness.PointerToValue()
		);
		BaseRoughnessTexture->GenerateMipMaps();
		OldResourceManager::Load(L"BaseRoughnessTexture", BaseRoughnessTexture);
		Texture2D * BaseNormalTexture = new Texture2D(
			IntVector2(BaseNormal.GetWidth(), BaseNormal.GetHeight()),
			CF_RGB,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_RGB,
			BaseNormal.PointerToValue()
		);
		BaseNormalTexture->GenerateMipMaps();
		OldResourceManager::Load(L"BaseNormalTexture", BaseNormalTexture);

		////
		Texture2D * FlamerAlbedoTexture = new Texture2D(
			IntVector2(FlamerAlbedo.GetWidth(), FlamerAlbedo.GetHeight()),
			CF_RGBA,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_RGBA,
			FlamerAlbedo.PointerToValue()
		);
		FlamerAlbedoTexture->GenerateMipMaps();
		OldResourceManager::Load(L"FlamerAlbedoTexture", FlamerAlbedoTexture);
		Texture2D * FlamerMetallicTexture = new Texture2D(
			IntVector2(FlamerMetallic.GetWidth(), FlamerMetallic.GetHeight()),
			CF_Red,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_Red,
			FlamerMetallic.PointerToValue()
		);
		FlamerMetallicTexture->GenerateMipMaps();
		OldResourceManager::Load(L"FlamerMetallicTexture", FlamerMetallicTexture);
		Texture2D * FlamerRoughnessTexture = new Texture2D(
			IntVector2(FlamerRoughness.GetWidth(), FlamerRoughness.GetHeight()),
			CF_Red,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_Red,
			FlamerRoughness.PointerToValue()
		);
		FlamerRoughnessTexture->GenerateMipMaps();
		OldResourceManager::Load(L"FlamerRoughnessTexture", FlamerRoughnessTexture);
		Texture2D * FlamerNormalTexture = new Texture2D(
			IntVector2(FlamerNormal.GetWidth(), FlamerNormal.GetHeight()),
			CF_RGB,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_RGB,
			FlamerNormal.PointerToValue()
		);
		FlamerNormalTexture->GenerateMipMaps();
		OldResourceManager::Load(L"FlamerNormalTexture", FlamerNormalTexture);
		////

		Texture2D * WhiteTexture = new Texture2D(
			IntVector2(White.GetWidth(), White.GetHeight()),
			CF_RGB,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_RGB,
			White.PointerToValue()
		);
		WhiteTexture->GenerateMipMaps();
		OldResourceManager::Load(L"WhiteTexture", WhiteTexture);
		Texture2D * BlackTexture = new Texture2D(
			IntVector2(Black.GetWidth(), Black.GetHeight()),
			CF_RGB,
			FM_MinMagLinear,
			SAM_Repeat,
			CF_RGB,
			Black.PointerToValue()
		);
		BlackTexture->GenerateMipMaps();
		OldResourceManager::Load(L"BlackTexture", BlackTexture);

		TextGenerator.GenerateGlyphAtlas(FontAtlas);

		FontMap = new Texture2D(
			IntVector2(TextGenerator.AtlasSize),
			CF_Red,
			FM_MinMagLinear,
			SAM_Border,
			CF_Red,
			FontAtlas.PointerToValue()
		);
		FontMap->GenerateMipMaps();

		// --- Create and compile our GLSL shader programs from text files
		Resource<ShaderProgram> * EquiToCubemapShader = OldResourceManager::Load<ShaderProgram>(L"EquirectangularToCubemap");
		Resource<ShaderProgram> * HDRClampingShader = OldResourceManager::Load<ShaderProgram>(L"HDRClampingShader");
		Resource<ShaderProgram> * BRDFShader = OldResourceManager::Load<ShaderProgram>(L"BRDFShader");
		Resource<ShaderProgram> * UnlitShader = OldResourceManager::Load<ShaderProgram>(L"UnLitShader");
		Resource<ShaderProgram> * RenderTextureShader = OldResourceManager::Load<ShaderProgram>(L"RenderTextureShader");
		Resource<ShaderProgram> * IntegrateBRDFShader = OldResourceManager::Load<ShaderProgram>(L"IntegrateBRDFShader");
		Resource<ShaderProgram> * RenderTextShader = OldResourceManager::Load<ShaderProgram>(L"RenderTextShader");
		Resource<ShaderProgram> * RenderCubemapShader = OldResourceManager::Load<ShaderProgram>(L"RenderCubemapShader");

		BaseMaterial.SetShaderProgram(BRDFShader->GetData());

		GetRenderPipeline().GetStage(L"TestStage")->CurrentMaterial = &BaseMaterial;

		UnlitMaterial.SetShaderProgram(UnlitShader->GetData());

		UnlitMaterialWire.SetShaderProgram(UnlitShader->GetData());
		UnlitMaterialWire.FillMode = FM_Wireframe;
		UnlitMaterialWire.CullMode = CM_None;

		RenderTextureMaterial.DepthFunction = DF_Always;
		RenderTextureMaterial.CullMode = CM_None;
		RenderTextureMaterial.SetShaderProgram(RenderTextureShader->GetData());

		RenderTextMaterial.DepthFunction = DF_Always;
		RenderTextMaterial.CullMode = CM_None;
		RenderTextMaterial.SetShaderProgram(RenderTextShader->GetData());

		RenderCubemapMaterial.CullMode = CM_None;
		RenderCubemapMaterial.SetShaderProgram(RenderCubemapShader->GetData());

		IntegrateBRDFMaterial.DepthFunction = DF_Always;
		IntegrateBRDFMaterial.CullMode = CM_None;
		IntegrateBRDFMaterial.SetShaderProgram(IntegrateBRDFShader->GetData());

		HDRClampingMaterial.DepthFunction = DF_Always;
		HDRClampingMaterial.CullMode = CM_None;
		HDRClampingMaterial.SetShaderProgram(HDRClampingShader->GetData());

		MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/SphereUV.obj"), true, [this](MeshLoader::FileData & ModelData) {
			if (ModelData.bLoaded) {
				SphereModel = Mesh(&(ModelData.Meshes.back()));
				SphereModel.SetUpBuffers();
			}
		});
		MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/Arrow.fbx"), false, [this](MeshLoader::FileData & ModelData) {
			for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
				LightModels.push_back(Mesh(&(*Data)));
				LightModels.back().SetUpBuffers();
			}
		});
		MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/Sponza.obj"), true, [this](MeshLoader::FileData & ModelData) {
			for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
				SceneModels.push_back(Mesh(&(*Data)));
				SceneModels.back().SetUpBuffers();
			}
		});
		// MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/EscafandraMV1971.fbx"), true, [&SceneModels](MeshLoader::FileData & ModelData) {
		// 	for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
		// 		SceneModels.push_back(Mesh(&(*Data)));
		// 		SceneModels.back().SetUpBuffers();
		// 	}
		// });

		Texture2D RenderedTexture = Texture2D(
			IntVector2(GetWindow().GetWidth(), GetWindow().GetHeight()) / 2, CF_RGBA32F, FM_MinLinearMagNearest, SAM_Repeat
		);

		///////// Create Matrices Buffer //////////////
		glGenBuffers(1, &ModelMatrixBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);

		EquirectangularTextureHDR = new Texture2D(
			IntVector2(Equirectangular.GetWidth(), Equirectangular.GetHeight()), CF_RGB16F, FM_MinMagLinear, SAM_Repeat
		);
		{
			RenderTarget Renderer = RenderTarget();
			Renderer.SetUpBuffers();
			EquirectangularTextureHDR->Use();
			HDRClampingMaterial.Use();
			HDRClampingMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
			HDRClampingMaterial.SetTexture2D("_EquirectangularMap", &EquirectangularTexture, 0);
			Renderer.Resize(EquirectangularTextureHDR->GetWidth(), EquirectangularTextureHDR->GetHeight());
			MeshPrimitives::Quad.BindVertexArray();
			Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
			HDRClampingMaterial.SetAttribMatrix4x4Array(
				"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
			);

			Renderer.PrepareTexture(EquirectangularTextureHDR);
			Renderer.Clear();
			MeshPrimitives::Quad.DrawInstanciated(1);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			Renderer.Delete();
			EquirectangularTextureHDR->GenerateMipMaps();
		}

		Texture2D * BRDFLut = new Texture2D(IntVector2(512), CF_RG16F, FM_MinMagLinear, SAM_Clamp);
		{
			RenderTarget Renderer = RenderTarget();
			Renderer.SetUpBuffers();
			Renderer.Use();
			IntegrateBRDFMaterial.Use();
			IntegrateBRDFMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
			Renderer.Resize(BRDFLut->GetWidth(), BRDFLut->GetHeight());
			MeshPrimitives::Quad.BindVertexArray();
			Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
			IntegrateBRDFMaterial.SetAttribMatrix4x4Array(
				"_iModelMatrix", 1, QuadPosition.PointerToValue(), ModelMatrixBuffer
			);

			Renderer.PrepareTexture(BRDFLut);
			Renderer.Clear();
			MeshPrimitives::Quad.DrawInstanciated(1);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			Renderer.Delete();
			// BRDFLut.GenerateMipMaps();
		}
		OldResourceManager::Load(L"BRDFLut", BRDFLut);

		CubemapTexture = new Cubemap(Equirectangular.GetHeight() / 2, CF_RGB16F, FM_MinMagLinear, SAM_Clamp);
		Cubemap::FromHDREquirectangular(*CubemapTexture, EquirectangularTextureHDR, EquiToCubemapShader->GetData());
		OldResourceManager::Load(L"CubemapTexture", CubemapTexture);

		Transforms.push_back(Transform());

		GetRenderPipeline().Initialize();
		GetRenderPipeline().ContextInterval(0);
	}

	virtual void OnUpdate() override {

		ProjectionMatrix = Matrix4x4::Perspective(
			60.0F * MathConstants::DegreeToRad,	// Aperute angle
			GetWindow().GetAspectRatio(),	    // Aspect ratio
			0.03F,						        // Near plane
			1000.0F						        // Far plane
		);

		// --- Camera rotation, position Matrix
		CursorPosition = Input::GetMousePosition();

		FrameRotation = Quaternion::EulerAngles(Vector3(CursorPosition.y, -CursorPosition.x));

		if (Input::IsKeyDown(SDL_SCANCODE_W)) {
			Vector3 Forward = FrameRotation * Vector3(0, 0, ViewSpeed);
			EyePosition += Forward * Time::GetDeltaTime() *
				(!Input::IsKeyDown(SDL_SCANCODE_LSHIFT) ? !Input::IsKeyDown(SDL_SCANCODE_LCTRL) ? 1.F : .1F : 4.F);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_A)) {
			Vector3 Right = FrameRotation * Vector3(ViewSpeed, 0, 0);
			EyePosition += Right * Time::GetDeltaTime() *
				(!Input::IsKeyDown(SDL_SCANCODE_LSHIFT) ? !Input::IsKeyDown(SDL_SCANCODE_LCTRL) ? 1.F : .1F : 4.F);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_S)) {
			Vector3 Back = FrameRotation * Vector3(0, 0, -ViewSpeed);
			EyePosition += Back * Time::GetDeltaTime() *
				(!Input::IsKeyDown(SDL_SCANCODE_LSHIFT) ? !Input::IsKeyDown(SDL_SCANCODE_LCTRL) ? 1.F : .1F : 4.F);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_D)) {
			Vector3 Left = FrameRotation * Vector3(-ViewSpeed, 0, 0);
			EyePosition += Left * Time::GetDeltaTime() *
				(!Input::IsKeyDown(SDL_SCANCODE_LSHIFT) ? !Input::IsKeyDown(SDL_SCANCODE_LCTRL) ? 1.F : .1F : 4.F);
		}

		CameraRayDirection = {
			(2.F * Input::GetMouseX()) / GetWindow().GetWidth() - 1.F,
			1.F - (2.F * Input::GetMouseY()) / GetWindow().GetHeight(),
			-1.F,
		};
		CameraRayDirection = ProjectionMatrix.Inversed() * CameraRayDirection;
		CameraRayDirection.z = -1.F;
		CameraRayDirection = ViewMatrix.Inversed() * CameraRayDirection;
		CameraRayDirection.Normalize();

		ViewMatrix = Transform(EyePosition, FrameRotation).GetGLViewMatrix();
		// ViewMatrix = Matrix4x4::Scaling(Vector3(1, 1, -1)).Inversed() * Matrix4x4::LookAt(EyePosition, EyePosition + FrameRotation * Vector3(0, 0, 1), FrameRotation * Vector3(0, 1)).Inversed();
		// Debug::Log(Debug::LogDebug, L"%ls", Text::FormatMath(ViewMatrix).c_str());

		if (Input::IsKeyDown(SDL_SCANCODE_N)) {
			MaterialMetalness -= 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_M)) {
			MaterialMetalness += 1.F * Time::GetDeltaTime();
			MaterialMetalness = std::clamp(MaterialMetalness, 0.F, 1.F);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_E)) {
			MaterialRoughness -= 0.5F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_R)) {
			MaterialRoughness += 0.5F * Time::GetDeltaTime();
			MaterialRoughness = std::clamp(MaterialRoughness, 0.F, 1.F);
		}
		if (Input::IsKeyDown(SDL_SCANCODE_L)) {
			LightIntencity += LightIntencity * Time::GetDeltaTime();
		}
		if (Input::IsKeyDown(SDL_SCANCODE_K)) {
			LightIntencity -= LightIntencity * Time::GetDeltaTime();
		}

		if (Input::IsKeyDown(SDL_SCANCODE_LSHIFT)) {
			if (Input::IsKeyDown(SDL_SCANCODE_W)) {
				if (BaseMaterial.FillMode == FM_Solid) {
					BaseMaterial.FillMode = FM_Wireframe;
				}
				else {
					BaseMaterial.FillMode = FM_Solid;
				}
			}
		}

		if (Input::IsKeyDown(SDL_SCANCODE_UP)) {
			MeshSelector += Time::GetDeltaTime() * 10;
			MeshSelector = MeshSelector > SceneModels.size() - 1 ? SceneModels.size() - 1 : MeshSelector;
		}
		if (Input::IsKeyDown(SDL_SCANCODE_DOWN)) {
			MeshSelector -= Time::GetDeltaTime() * 10;
			MeshSelector = MeshSelector < 0 ? 0 : MeshSelector;
		}
		if (Input::IsKeyDown(SDL_SCANCODE_RIGHT)) {
			MultiuseValue += Time::GetDeltaTime() * MultiuseValue;
		}
		if (Input::IsKeyDown(SDL_SCANCODE_LEFT)) {
			MultiuseValue -= Time::GetDeltaTime() * MultiuseValue;
		}

		if (Input::IsKeyDown(SDL_SCANCODE_LSHIFT)) {
			if (Input::IsKeyDown(SDL_SCANCODE_I)) {
				FontSize += Time::GetDeltaTime() * FontSize;
			}

			if (Input::IsKeyDown(SDL_SCANCODE_K)) {
				FontSize -= Time::GetDeltaTime() * FontSize;
			}
		}
		else {
			if (Input::IsKeyDown(SDL_SCANCODE_I)) {
				FontBoldness += Time::GetDeltaTime() / 10;
			}

			if (Input::IsKeyDown(SDL_SCANCODE_K)) {
				FontBoldness -= Time::GetDeltaTime() / 10;
			}
		}

		if (Input::IsKeyDown(SDL_SCANCODE_V)) {
			for (int i = 0; i < 10; i++) {
				RenderingText[1] += (unsigned long)(rand() % 0x3ff);
			}
		}

		if (Input::IsKeyDown(SDL_SCANCODE_SPACE)) {
			TestArrowTransform.Position = EyePosition;
			TestArrowDirection = CameraRayDirection;
			TestArrowTransform.Rotation = Quaternion::LookRotation(CameraRayDirection, Vector3(0, 1, 0));
		}

		for (int i = 0; i < TextCount; i++) {
			if (TextGenerator.FindCharacters(RenderingText[i]) > 0) {
				TextGenerator.GenerateGlyphAtlas(FontAtlas);
				FontMap->Delete();
				delete FontMap;
				FontMap = new Texture2D(
					IntVector2(TextGenerator.AtlasSize),
					CF_Red,
					FM_MinMagLinear,
					SAM_Border,
					CF_Red,
					FontAtlas.PointerToValue()
				);
				FontMap->GenerateMipMaps();
			}
		}

		// Transforms[0].Rotation = Quaternion::AxisAngle(Vector3(0, 1, 0).Normalized(), Time::GetDeltaTime() * 0.04F) * Transforms[0].Rotation;
		TransformMat = Transforms[0].GetLocalToWorldMatrix();
		InverseTransform = Transforms[0].GetWorldToLocalMatrix();

		TestArrowTransform.Scale = 0.1F;
		TestArrowTransform.Position += TestArrowDirection * TestSphereVelocity * Time::GetDeltaTime();
	}

	virtual void OnRender() override {
		GetRenderPipeline().PrepareFrame();

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
							SceneModels[MeshCount].Data.Vertices[(*Face)[2]].Position, BaseMaterial.CullMode != CM_CounterClockWise
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
							Hits[0].Normal.Normalize();
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
			Vector3(Input::GetMousePosition())
		).ToMatrix4x4() * Matrix4x4::Translation(Vector3(0.5F))) * Vector4(0.F, 0.F, 0.F, 1.F), 0.5F));
		// bTestResult = RTRenderToTexture2D(&RenderedTexture, &Spheres, curandomStateArray);

		// Framebuffer.Use();

		RenderCubemapMaterial.Use();

		float MaterialRoughnessTemp = (1 - MaterialRoughness) * (CubemapTexture->GetMipmapCount() - 3);
		RenderCubemapMaterial.SetMatrix4x4Array("_ProjectionMatrix", ProjectionMatrix.PointerToValue());
		RenderCubemapMaterial.SetMatrix4x4Array("_ViewMatrix", ViewMatrix.PointerToValue());
		RenderCubemapMaterial.SetTextureCubemap("_Skybox", CubemapTexture, 0);
		RenderCubemapMaterial.SetFloat1Array("_Roughness", &MaterialRoughnessTemp);

		if (SphereModel.Data.Faces.size() >= 1) {
			SphereModel.SetUpBuffers();
			SphereModel.BindVertexArray();

			Matrix4x4 MatrixScale = Matrix4x4::Scaling({ 500, 500, 500 });
			RenderCubemapMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, MatrixScale.PointerToValue(), ModelMatrixBuffer);
			SphereModel.DrawElement();
		}

		GetRenderPipeline().GetStage(L"TestStage")->SetEyeTransform(Transform(EyePosition, FrameRotation));
		GetRenderPipeline().GetStage(L"TestStage")->SetViewProjection(Matrix4x4::Perspective(
			60.0F * MathConstants::DegreeToRad,	 // Aperute angle
			GetWindow().GetAspectRatio(),		 // Aspect ratio
			0.03F,								 // Near plane
			1000.0F								 // Far plane
		));

		GetRenderPipeline().RunStage(L"TestStage");

		BaseMaterial.Use();

		BaseMaterial.SetFloat3Array("_ViewPosition", EyePosition.PointerToValue());
		BaseMaterial.SetFloat3Array("_Lights[0].Position", LightPosition0.PointerToValue());
		BaseMaterial.SetFloat3Array("_Lights[0].Color", Vector3(1.F, 1.F, .9F).PointerToValue());
		BaseMaterial.SetFloat1Array("_Lights[0].Intencity", &LightIntencity);
		BaseMaterial.SetFloat3Array("_Lights[1].Position", LightPosition1.PointerToValue());
		BaseMaterial.SetFloat3Array("_Lights[1].Color", Vector3(1.F, 1.F, .9F).PointerToValue());
		BaseMaterial.SetFloat1Array("_Lights[1].Intencity", &LightIntencity);
		BaseMaterial.SetFloat1Array("_Material.Metalness", &MaterialMetalness);
		BaseMaterial.SetFloat1Array("_Material.Roughness", &MaterialRoughness);
		BaseMaterial.SetFloat3Array("_Material.Color", Vector3(1.F).PointerToValue());

		BaseMaterial.SetMatrix4x4Array("_ProjectionMatrix", ProjectionMatrix.PointerToValue());
		BaseMaterial.SetMatrix4x4Array("_ViewMatrix", ViewMatrix.PointerToValue());
		BaseMaterial.SetTexture2D("_MainTexture", OldResourceManager::Get<Texture2D>(L"BaseAlbedoTexture")->GetData(), 0);
		BaseMaterial.SetTexture2D("_NormalTexture", OldResourceManager::Get<Texture2D>(L"BaseNormalTexture")->GetData(), 1);
		BaseMaterial.SetTexture2D("_RoughnessTexture", OldResourceManager::Get<Texture2D>(L"BaseRoughnessTexture")->GetData(), 2);
		BaseMaterial.SetTexture2D("_MetallicTexture", OldResourceManager::Get<Texture2D>(L"BaseMetallicTexture")->GetData(), 3);
		BaseMaterial.SetTexture2D("_BRDFLUT", OldResourceManager::Get<Texture2D>(L"BRDFLut")->GetData(), 4);
		BaseMaterial.SetTextureCubemap("_EnviromentMap", CubemapTexture, 5);
		float CubemapTextureMipmaps = CubemapTexture->GetMipmapCount();
		BaseMaterial.SetFloat1Array("_EnviromentMapLods", &CubemapTextureMipmaps);

		// Transforms[0].Position += Transforms[0].Rotation * Vector3(0, 0, Time::GetDeltaTime() * 2);

		size_t TotalHitCount = 0;
		Ray CameraRay(EyePosition, CameraRayDirection);
		for (int MeshCount = (int)MeshSelector; MeshCount >= 0 && MeshCount < (int)SceneModels.size(); ++MeshCount) {
			const MeshData & ModelData = SceneModels[MeshCount].Data;
			BoundingBox3D ModelSpaceAABox = ModelData.Bounding.Transform(TransformMat);
			TArray<RayHit> Hits;

			if (Physics::RaycastAxisAlignedBox(CameraRay, ModelSpaceAABox)) {
				RayHit Hit;
				Ray ModelSpaceCameraRay(
					InverseTransform.MultiplyPoint(EyePosition),
					InverseTransform.MultiplyVector(CameraRayDirection)
				);
				for (MeshFaces::const_iterator Face = ModelData.Faces.begin(); Face != ModelData.Faces.end(); ++Face) {
					if (Physics::RaycastTriangle(
						Hit, ModelSpaceCameraRay,
						ModelData.Vertices[(*Face)[0]].Position,
						ModelData.Vertices[(*Face)[1]].Position,
						ModelData.Vertices[(*Face)[2]].Position, BaseMaterial.CullMode != CM_CounterClockWise
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
						Hits[0].Normal.Normalize();
						Vector3 ReflectedCameraDir = Vector3::Reflect(CameraRayDirection, Hits[0].Normal);
						Matrix4x4 HitMatrix[2] = {
							Matrix4x4::Translation(CameraRay.PointAt(Hits[0].Stamp)) *
							Matrix4x4::Rotation(Quaternion::LookRotation(ReflectedCameraDir, Vector3(0, 1, 0))) *
							Matrix4x4::Scaling(0.1F),
							Matrix4x4::Translation(CameraRay.PointAt(Hits[0].Stamp)) *
							Matrix4x4::Rotation(Quaternion::LookRotation(Hits[0].Normal, Vector3(0, 1, 0))) *
							Matrix4x4::Scaling(0.07F)
						};
						BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 2, &HitMatrix[0], ModelMatrixBuffer);

						LightModels[0].DrawInstanciated(2);
						TriangleCount += LightModels[0].Data.Faces.size() * 1;
						VerticesCount += LightModels[0].Data.Vertices.size() * 1;
					}
				}
			}

			SceneModels[MeshCount].SetUpBuffers();
			SceneModels[MeshCount].BindVertexArray();

			BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &TransformMat, ModelMatrixBuffer);
			SceneModels[MeshCount].DrawInstanciated(1);

			TriangleCount += ModelData.Faces.size() * 1;
			VerticesCount += ModelData.Vertices.size() * 1;
		}

		if (LightModels.size() > 0) {
			LightModels[0].SetUpBuffers();
			LightModels[0].BindVertexArray();

			Matrix4x4 ModelMatrix = TestArrowTransform.GetLocalToWorldMatrix();
			BaseMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1, &ModelMatrix, ModelMatrixBuffer);

			LightModels[0].DrawInstanciated(1);
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

		UnlitMaterial.SetMatrix4x4Array("_ProjectionMatrix", ProjectionMatrix.PointerToValue());
		UnlitMaterial.SetMatrix4x4Array("_ViewMatrix", ViewMatrix.PointerToValue());
		UnlitMaterial.SetFloat3Array("_ViewPosition", EyePosition.PointerToValue());
		UnlitMaterial.SetFloat4Array("_Material.Color", Vector4(LightIntencity, 0, 0, 1).PointerToValue());

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
		glViewport(0, 0, EquirectangularTextureHDR->GetWidth() / 4 * abs(1 - MultiuseValue), EquirectangularTextureHDR->GetHeight() / 4 * abs(1 - MultiuseValue));

		RenderTextureMaterial.Use();

		RenderTextureMaterial.SetFloat1Array("_Time", &AppTime);
		RenderTextureMaterial.SetFloat2Array("_MainTextureSize", EquirectangularTextureHDR->GetDimension().FloatVector2().PointerToValue());
		RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
		RenderTextureMaterial.SetTexture2D("_MainTexture", EquirectangularTextureHDR, 0);
		float LodLevel = log2f((float)EquirectangularTextureHDR->GetWidth()) * abs(MultiuseValue);
		RenderTextureMaterial.SetFloat1Array("_Lod", &LodLevel);

		MeshPrimitives::Quad.BindVertexArray();

		Matrix4x4 QuadPosition = Matrix4x4::Translation({ 0, 0, 0 });
		RenderTextureMaterial.SetAttribMatrix4x4Array("_iModelMatrix", 1,
			/*(Quaternion({ MathConstants::HalfPi, 0, 0}).ToMatrix4x4() * */QuadPosition.PointerToValue(),
			ModelMatrixBuffer
		);

		MeshPrimitives::Quad.DrawInstanciated(1);

		glViewport(0, 0, GetWindow().GetWidth(), GetWindow().GetHeight());
		// --- Activate corresponding render state
		RenderTextMaterial.Use();
		RenderTextMaterial.SetFloat1Array("_Time", &AppTime);
		RenderTextMaterial.SetFloat2Array("_MainTextureSize", FontMap->GetDimension().FloatVector2().PointerToValue());
		RenderTextMaterial.SetMatrix4x4Array("_ProjectionMatrix",
			Matrix4x4::Orthographic(0.F, (float)GetWindow().GetWidth(), 0.F, (float)GetWindow().GetHeight()).PointerToValue()
		);

		float FontScale = (FontSize / TextGenerator.GlyphHeight);
		RenderTextMaterial.SetTexture2D("_MainTexture", FontMap, 0);
		RenderTextMaterial.SetFloat1Array("_TextSize", &FontScale);
		RenderTextMaterial.SetFloat1Array("_TextBold", &FontBoldness);

		double TimeCount = 0;
		int TotalCharacterSize = 0;
		DynamicMesh.Clear();
		for (int i = 0; i < TextCount; i++) {
			Timer.Start();
			Vector2 Pivot = TextPivot + Vector2(0.F, GetWindow().GetHeight() - (i + 1) * FontSize + FontSize / TextGenerator.GlyphHeight);
			TextGenerator.GenerateMesh(
				Box2D(0, 0, (float)GetWindow().GetWidth(), Pivot.y),
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
			L"Character(%.2f μs, %d), Temp [%.1f°], %.1f FPS (%.2f ms), LightIntensity(%.3f), Vertices(%ls), Cursor(%ls), Camera(P%ls, R%ls)",
			TimeCount / double(TotalCharacterSize) * 1000.0,
			TotalCharacterSize,
			Debug::GetDeviceTemperature(0),
			Time::GetFrameRatePerSecond(),
			(1.F / Time::GetFrameRatePerSecond()) * 1000.F,
			LightIntencity / 10000.F + 1.F,
			Text::FormatUnit(VerticesCount, 2).c_str(),
			Text::FormatMath(CursorPosition).c_str(),
			Text::FormatMath(EyePosition).c_str(),
			Text::FormatMath(Math::ClampAngleComponents(FrameRotation.ToEulerAngles())).c_str()
		);

		if (Input::IsKeyDown(SDL_SCANCODE_ESCAPE)) {
			ShouldClose();
		}

	}

	virtual void OnTerminate() override {
		if (Space::GetMainSpace() == NULL) {
			return;
		}

		Space::Destroy(Space::GetMainSpace());
	}

public:
	typedef Application Supper;
	
	SandboxApplication() : Supper() {}
};

EmptySource::Application * EmptySource::CreateApplication() {
	return new SandboxApplication();
}
