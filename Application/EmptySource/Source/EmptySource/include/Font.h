#pragma once

#include "..\include\Glyph.h"

class Font {
private:
	static struct FT_LibraryRec_ * FreeTypeLibrary;


public:

	struct FT_FaceRec_ * Face;
	
	//* Initialize FreeType Library
	static bool InitializeFreeType();

	//* Initialize Front
	void Initialize(struct FileStream * File);

	//* Get the index in font of given character
	unsigned int GetGlyphIndex(const unsigned long & Character) const;

	//* Set the glyph height for rasterization
	void SetGlyphHeight(const unsigned int& Size) const;

	//* Clear all the FreeType info
	void Clear();

	//* Default Constructor
	Font();
};