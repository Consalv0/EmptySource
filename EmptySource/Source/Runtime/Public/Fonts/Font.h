#pragma once

#include "Fonts/Glyph.h"

namespace ESource {

	struct FT_Context {
		Point2 position;
		Shape2D *shape;
		Shape2DContour *contour;
	};

	class Font {
	private:
		static struct FT_LibraryRec_ * FreeTypeLibrary;

		struct FT_FaceRec_ * Face;

	public:

		//* Default Constructor
		Font();

		//* Initialize FreeType Library
		static bool InitializeFreeType();

		//* Initialize Front
		void Initialize(struct FileStream * File);

		//* Get the index in font of given character
		uint32_t GetGlyphIndex(const uint32_t & Character) const;

		//* Get a copy of a Glyph
		bool GetGlyph(FontGlyph & Glyph, const uint32_t& GlyphIndex);

		//* Set the glyph height for rasterization
		void SetGlyphHeight(const uint32_t& Size) const;

		//* Clear all the FreeType info
		void Clear();
	};

}