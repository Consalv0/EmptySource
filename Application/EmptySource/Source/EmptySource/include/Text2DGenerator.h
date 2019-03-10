#pragma once

#include "..\include\Font.h"
#include "..\include\Mesh.h"
#include "..\include\Text.h"

struct Text2DGenerator {
private:

	TDictionary<unsigned long, TextGlyph *> LoadedCharacters;

public:
	Font * TextFont;
	int GlyphHeight;
	int AtlasSize;
	int RenderScale;
	Vector2 Pivot;

	//* Prepare Character Info
	void PrepareCharacters(const unsigned long & From, const unsigned long & To);

	//* Generate mesh geometry for rasterization. 
	//* The uv of the quads are calculated in the gryphs creation 
	void GenerateTextMesh(Vector2 Pivot, const WString & InText, MeshFaces * Faces, MeshVertices * Vertices);

	//* Clear generator memory
	void Clear();

	//* Prepare Texture Data
	unsigned char * GenerateTextureAtlas();
};