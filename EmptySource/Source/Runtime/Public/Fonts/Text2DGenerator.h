#pragma once

#include "CoreTypes.h"
#include "Fonts/Font.h"
#include "Rendering/Mesh.h"
#include "Rendering/PixelMap.h"

namespace ESource {

	struct Text2DGenerator {
	private:
		TDictionary<unsigned long, FontGlyph *> LoadedCharacters;

	public:
		Font * TextFont = NULL;
		int GlyphHeight = 24;
		int AtlasSize = 512;
		float PixelRange = 2.F;
		Vector2 Pivot = 0;

		//* Prepare Character Info
		void PrepareCharacters(const WChar* Characters, const size_t & Count);

		//* Prepare Character Info
		void PrepareCharacters(const uint32_t & From, const uint32_t & To);

		//* Update needed characters (returns the number of characters added)
		int PrepareCharacters(const WString & InText);

		//* Generate mesh geometry for rasterization. 
		void GenerateMesh(const Box2D & Box, float HeightSize, bool RightHanded, WString InText, MeshFaces * Faces, MeshVertices * Vertices);

		bool IsCharacterLoaded(unsigned long Character) const;

		//* Precalculate the leght of the text rendered.
		Vector2 GetLenght(float HeightSize, const WString & InText);

		//* Clear generator memory
		void Clear();

		//* Prepare Texture Data
		//* The UV of the glyph quads are calculated here
		bool GenerateGlyphAtlas(PixelMap & Atlas);

	private:

		void AddNewGlyph(const FontGlyph & Glyph);

	};

}