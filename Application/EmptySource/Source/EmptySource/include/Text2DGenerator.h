#pragma once

#include "..\include\Font.h"
#include "..\include\Mesh.h"
#include "..\include\Text.h"
#include "..\include\Bitmap.h"

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
	void PrepareCharacters(const unsigned long & From, const unsigned long & To);

	//* Generate mesh geometry for rasterization. 
	//* The uv of the quads are calculated in the gryphs creation 
	void GenerateMesh(Vector2 Pivot, float HeightSize, const WString & InText, MeshFaces * Faces, MeshVertices * Vertices);

	//* Clear generator memory
	void Clear();

	//* Prepare Texture Data
	bool GenerateGlyphAtlas(Bitmap<unsigned char> & Atlas);
};