
#include "CoreMinimal.h"
#include "Resources/TextureResource.h"
#include "Resources/TextureManager.h"
#include "Resources/ImageConversion.h"

#include "Rendering/RenderTarget.h"
#include "Rendering/MeshPrimitives.h"
#include "Rendering/Material.h"

#include "Utility/TextFormattingMath.h"

namespace ESource {
	
	RTexture::~RTexture() {
		Unload();
	}

	bool RTexture::IsValid() const {
		return LoadState == LS_Loaded && TexturePointer->IsValid();
	}

	void RTexture::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			LOG_CORE_DEBUG(L"Loading Texture '{}'...", Name.GetDisplayName().c_str());
			if (!Origin.empty()) {
				FileStream * TextureFile = FileManager::GetFile(Origin);
				if (TextureFile == NULL) {
					LOG_CORE_ERROR(L"Error reading file for texture: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}

				ImageConversion::PixelMapInfo Info;
				ImageConversion::ParsingOptions Options = { TextureFile, ColorFormat, bFlipVerticaly };
				if (!ImageConversion::Load(Info, Options)) {
					LOG_CORE_ERROR(L"Error reading file for texture: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}
				else {
					Pixels = Info.Pixels;
					Size = Pixels.GetSize();
				}
				TextureFile->Close();
			}

			switch (Dimension) {
			case ETextureDimension::Texture2D:
				if (Pixels.IsEmpty())
					TexturePointer = Texture2D::Create(Size, ColorFormat, FilterMode, AddressMode);
				else
					TexturePointer = Texture2D::Create(Size, ColorFormat, FilterMode, AddressMode,
						Pixels.GetColorFormat(), Pixels.PointerToValue());
				break;
			case ETextureDimension::Cubemap:
				TexturePointer = Cubemap::Create(Size.X, ColorFormat, FilterMode, AddressMode);
				Size = { Size.X, Size.X, 6 };
				break;
			case ETextureDimension::Texture1D:
			case ETextureDimension::Texture3D:
			default:
				break;
			}
		}

		if (Size.MagnitudeSquared() == 0) {
			LOG_CORE_CRITICAL(L"Assigned Invalid Size in Texture '{}' : {}", Name.GetDisplayName().c_str(), Text::FormatMath(Size).c_str());
		}
		LoadState = TexturePointer != NULL ? LS_Loaded : LS_Unloaded;
		if (bBuildMipMapsOnLoad) GenerateMipMaps();
		if (!bConservePixelMapOnLoad) ClearPixelData();
	}

	void RTexture::LoadAsync() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			LOG_CORE_DEBUG(L"Loading Texture '{}'...", Name.GetDisplayName().c_str());
			if (!Origin.empty()) {
				FileStream * TextureFile = FileManager::GetFile(Origin);
				if (TextureFile == NULL) {
					LOG_CORE_ERROR(L"Error reading file for texture: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}

				ImageConversion::ParsingOptions Options = { TextureFile, ColorFormat, bFlipVerticaly };
				ImageConversion::LoadAsync(Options, [this](ImageConversion::PixelMapInfo & Info) {
					Pixels = Info.Pixels;
					Size = Pixels.GetSize();

					switch (Dimension) {
					case ETextureDimension::Texture2D:
						if (Pixels.IsEmpty())
							TexturePointer = Texture2D::Create(Size, ColorFormat, FilterMode, AddressMode);
						else
							TexturePointer = Texture2D::Create(Size, ColorFormat, FilterMode, AddressMode,
								Pixels.GetColorFormat(), Pixels.PointerToValue());
						break;
					case ETextureDimension::Cubemap:
						TexturePointer = Cubemap::Create(Size.X, ColorFormat, FilterMode, AddressMode);
						Size = { Size.X, Size.X, 6 };
						break;
					case ETextureDimension::Texture1D:
					case ETextureDimension::Texture3D:
					default:
						break;
					}

					if (Size.MagnitudeSquared() == 0) {
						LOG_CORE_CRITICAL(L"Assigned Invalid Size in Texture '{}' : {}", Name.GetDisplayName().c_str(), Text::FormatMath(Size).c_str());
					}

					LoadState = TexturePointer != NULL ? LS_Loaded : LS_Unloaded;
					if (bBuildMipMapsOnLoad) GenerateMipMaps();
					if (!bConservePixelMapOnLoad) ClearPixelData();
				});
			}
		}
	}

	void RTexture::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		delete TexturePointer;
		TexturePointer = NULL;
		Pixels = PixelMap();
		MipMapCount = 0;
		LoadState = LS_Unloaded;
	}

	void RTexture::Reload() {
		Unload();
		Load();
	}

	uint32_t RTexture::GetMipMapCount() const {
		if (IsValid())
			return MipMapCount;
		return 0;
	}

	void RTexture::GenerateMipMaps() {
		if (IsValid() && MipMapCount <= 1) { 
			MipMapCount = (uint32_t)log2f((float)Size.X);
			TexturePointer->GenerateMipMaps(FilterMode, GetMipMapCount()); 
		}
	}

	void RTexture::DeleteMipMaps() {
		MipMapCount = 0;
	}

	void RTexture::SetGenerateMipMapsOnLoad(bool Option) {
		bBuildMipMapsOnLoad = Option;
	}

	void RTexture::SetFlipVertically(bool Option) {
		bFlipVerticaly = Option;
	}

	void RTexture::SetConservePixelMapOnLoad(bool Option) {
		bConservePixelMapOnLoad = Option;
	}

	void RTexture::SetSize(const IntVector3 & NewSize) {
		if (LoadState == LS_Unloaded) {
			Size = NewSize;
		}
	}

	void RTexture::SetPixelData(const PixelMap & Data) {
		if (LoadState == LS_Unloaded && Origin.empty()) {
			Size = Data.GetSize();
			Pixels = Data;
		}
	}

	void RTexture::ClearPixelData() {
		Pixels = PixelMap();
	}

	float RTexture::GetAspectRatio() const {
		return (float)Size.X / (float)Size.Y;
	}

	bool RTexture::RenderHDREquirectangular(RTexturePtr Equirectangular, Material * CubemapMaterial, bool bGenerateMipMaps) {
		if (!IsValid() || Dimension != ETextureDimension::Cubemap) return false;
		if (bGenerateMipMaps) GenerateMipMaps();
		
		static const Matrix4x4 CaptureProjection = Matrix4x4::Perspective(90.F * MathConstants::DegreeToRad, 1.F, 0.1F, 10.F);
		static const std::pair<ECubemapFace, Matrix4x4> CaptureViews[] = {
		   { ECubemapFace::Right, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Left,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3(-1.F,  0.F,  0.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Up,    Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F, -1.F,  0.F), Vector3(0.F,  0.F, -1.F)) },
		   { ECubemapFace::Down,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  1.F,  0.F), Vector3(0.F,  0.F,  1.F)) },
		   { ECubemapFace::Back,  Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F,  1.F), Vector3(0.F, -1.F,  0.F)) },
		   { ECubemapFace::Front, Matrix4x4::LookAt(Vector3(0.F, 0.F, 0.F), Vector3( 0.F,  0.F, -1.F), Vector3(0.F, -1.F,  0.F)) }
		};
		
		static RenderTargetPtr Renderer = RenderTarget::Create();
		
		// --- Convert HDR equirectangular environment map to cubemap equivalent
		CubemapMaterial->Use();
		CubemapMaterial->SetTexture2D("_EquirectangularMap", Equirectangular, 0);
		CubemapMaterial->SetMatrix4x4Array("_ProjectionMatrix", CaptureProjection.PointerToValue());
		
		const uint32_t MaxMipLevels = (uint32_t)GetMipMapCount();
		for (uint32_t Lod = 0; Lod <= MaxMipLevels; ++Lod) {
		
			float Roughness = (float)Lod / (float)(MaxMipLevels);
			CubemapMaterial->SetFloat1Array("_Roughness", &Roughness);
		
			Renderer->Bind();
		
			for (auto View : CaptureViews) {
				CubemapMaterial->SetMatrix4x4Array("_ViewMatrix", View.second.PointerToValue());
		
				MeshPrimitives::Cube.GetVertexArray()->Bind();
				
				Renderer->BindCubemapFace((Cubemap *)TexturePointer, Size.X, View.first, Lod);
				Rendering::SetViewport({ 0, 0, Size.X >> Lod, Size.X >> Lod });
				Renderer->Clear();

				Rendering::DrawIndexed(MeshPrimitives::Cube.GetVertexArray());
				if (!Renderer->CheckStatus()) return false;
			}
		}
		
		return true;
	}

	RTexture::RTexture(
		const IName & Name, const WString & Origin,
		ETextureDimension Dimension, EPixelFormat Format, EFilterMode FilterMode, ESamplerAddressMode AddressMode, const IntVector3& Size,
		bool MipMapsOnLoad, bool FlipVertically
	) 
		: ResourceHolder(Name, Origin), Dimension(Dimension), FilterMode(FilterMode), AddressMode(AddressMode), ColorFormat(Format), Size(Size) {
		bFlipVerticaly = FlipVertically;
		TexturePointer = NULL;
		MipMapCount = 1; bBuildMipMapsOnLoad = MipMapsOnLoad;
	}

}