#pragma once

#include "Resources/ResourceManager.h"
#include "Resources/ResourceHolder.h"
#include "Rendering/Texture.h"

namespace ESource {

	typedef std::shared_ptr<class RTexture> RTexturePtr;

	class RTexture : public ResourceHolder {
	public:
		~RTexture();

		virtual bool IsValid() const override;

		virtual void Load() override;

		virtual void LoadAsync() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }

		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Shader; }

		virtual inline size_t GetMemorySize() const override { return Pixels.GetMemorySize(); };

		inline Texture * GetTexture() { return TexturePointer; };

		inline ETextureDimension GetDimension() const { return Dimension; }

		uint32_t GetMipMapCount() const;

		void GenerateMipMaps();

		void DeleteMipMaps();

		void SetGenerateMipMapsOnLoad(bool Option);

		void SetFlipVertically(bool Option);

		void SetConservePixelMapOnLoad(bool Option);

		inline IntVector3 GetSize() const { return Size; }

		// Change the size of the texture, this will only be aplied if the texture is not loaded
		void SetSize(const IntVector3 & NewSize);

		// Change the pixels in texture, this will only be aplied if the texture is not loaded 
		// and its origin is empty
		void SetPixelData(const PixelMap & Data);

		void ClearPixelData();

		float GetAspectRatio() const;

		bool RenderHDREquirectangular(RTexturePtr Equirectangular, class Material * CubemapMaterial, bool GenerateMipMaps);

		static inline EResourceType GetType() { return EResourceType::RT_Shader; };

	protected:
		friend class TextureManager;

		RTexture(
			const IName & Name, const WString & Origin, ETextureDimension Dimension,
			EPixelFormat Format, EFilterMode FilterMode, ESamplerAddressMode AddressMode, const IntVector3 & Size = 0,
			bool MipMapsOnLoad = false, bool FlipVerticaly = true
		);

	private:
		Texture * TexturePointer;

		PixelMap Pixels;

		IntVector3 Size;

		ETextureDimension Dimension;

		EFilterMode FilterMode;

		ESamplerAddressMode AddressMode;

		EPixelFormat ColorFormat;

		uint32_t MipMapCount;

		bool bBuildMipMapsOnLoad;
		bool bConservePixelMapOnLoad;

		bool bFlipVerticaly;

	};

}
