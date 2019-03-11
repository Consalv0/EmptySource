
#include "..\External\ft2build.h"
#include FT_FREETYPE_H
#include "..\External\freetype\freetype.h"

#include "..\include\Core.h"
#include "..\include\Utility\LogFreeType.h"
#include "..\include\Utility\LogCore.h"
#include "..\include\Text2DGenerator.h"
#include "..\include\Utility\Timer.h"

void Text2DGenerator::PrepareCharacters(const unsigned long & From, const unsigned long & To) {
	TextFont->SetGlyphHeight(GlyphHeight);
	for (unsigned long Character = From; Character <= To; Character++) {
		unsigned int GlyphIndex = TextFont->GetGlyphIndex(Character);
		FT_Error Error;
		if (Error = FT_Load_Glyph(TextFont->Face, GlyphIndex, FT_LOAD_RENDER)) {
			Debug::Log(Debug::LogError, L"Failed to load Glyph '%c', %s", Character, FT_ErrorMessage(Error));
		}
		else {
			LoadedCharacters.insert_or_assign(Character, new TextGlyph(
				Character,
				{ (int)TextFont->Face->glyph->bitmap.width, (int)TextFont->Face->glyph->bitmap.rows },
				{ (int)TextFont->Face->glyph->bitmap_left, (int)TextFont->Face->glyph->bitmap_top },
				(int)TextFont->Face->glyph->advance.x
			));
			{
				TextGlyph * Glyph = LoadedCharacters[Character];
				LoadedCharacters[Character]->RasterizedData = new unsigned char[Glyph->Size.x * Glyph->Size.y];
				memcpy(Glyph->RasterizedData, TextFont->Face->glyph->bitmap.buffer, Glyph->Size.x * Glyph->Size.y);
			}
		}
	}
}

void Text2DGenerator::GenerateMesh(Vector2 Pivot, const WString & InText, MeshFaces * Faces, MeshVertices * Vertices) {
	
	if (InText.size() == 0) return;

	size_t InTextSize = InText.size();
	size_t InitialFacesSize = Faces->size();
	size_t InitialVerticesSize = Vertices->size();

	Faces->resize(InitialFacesSize + InTextSize * 2);
	Vertices->resize(InitialVerticesSize + InTextSize * 4);
	
	int VertexCount = 0;
	IntVector3 * TextFacesEnd = &Faces->at(InitialFacesSize);
	MeshVertex * TextVerticesEnd = &Vertices->at(InitialVerticesSize);

	// --- Iterate through all characters
	for (WString::const_iterator Character = InText.begin(); Character != InText.end(); Character++) {
		TextGlyph * Glyph = LoadedCharacters[*Character];
		if (Glyph == NULL) {
			Pivot.x += GlyphHeight / 2.F;
			continue;
		}

		Glyph->GetQuadMesh(Pivot, TextVerticesEnd);
		TextFacesEnd->x = VertexCount;
		TextFacesEnd->y = VertexCount + 1;
		(TextFacesEnd++)->z = VertexCount + 2;
		TextFacesEnd->x = VertexCount + 3;
		TextFacesEnd->y = VertexCount;
		(TextFacesEnd++)->z = VertexCount + 1;

		VertexCount += 4;
		TextVerticesEnd += 4;

		// --- Advance cursor for next glyph (note that advance is number of 1/32 pixels)
		Pivot.x += (Glyph->Advance >> 6);
	}

	Faces->resize(InitialFacesSize + VertexCount / 2);
	Vertices->resize(InitialVerticesSize + VertexCount);
}

void Text2DGenerator::Clear() {
	LoadedCharacters.clear();
}

unsigned char * Text2DGenerator::GenerateTextureAtlas() {
	int AtlasSizeSqr = AtlasSize * AtlasSize;
	unsigned char * AtlasData = new unsigned char[AtlasSizeSqr];

	for (int i = 0; i < AtlasSizeSqr; i++) {
		// if (i % GlyphHeight == 0 || (i / (AtlasSize)) % GlyphHeight == 0) {
		// 	AtlasData[i] = 255;
		// }
		// else {
			AtlasData[i] = 0;
		// }
	}

	IntVector2 AtlasPosition;
	for (TDictionary<unsigned long, TextGlyph*>::iterator Begin = LoadedCharacters.begin(); Begin != LoadedCharacters.end(); Begin++) {
		TextGlyph * Character = Begin->second;
		int IndexPos = AtlasPosition.x + AtlasPosition.y * AtlasSize;
		IndexPos += Character->Bearing.x;

		// --- Asign the current UV Position
		Character->MinU = (Character->Bearing.u + AtlasPosition.u) / (float)AtlasSize;
		Character->MaxU = (Character->Bearing.u + AtlasPosition.u + Character->Size.u) / (float)AtlasSize;
		Character->MinV = (AtlasPosition.v) / (float)AtlasSize;
		Character->MaxV = (AtlasPosition.v + Character->Size.v) / (float)AtlasSize;
		
		// --- If current position exceds the canvas with the next character, reset the position in Y
		AtlasPosition.y += GlyphHeight;
		if (AtlasPosition.y * AtlasSize > (AtlasSizeSqr - GlyphHeight * AtlasSize)) {
			AtlasPosition.x += GlyphHeight;
			AtlasPosition.y = 0;
		}
		
		// --- Render current character in the current position
		for (int i = Character->Size.y - 1; i >= 0; i--) {
			for (int j = 0; j < Character->Size.x; j++) {
				if (IndexPos >= AtlasSizeSqr) {
					IndexPos -= AtlasSizeSqr / GlyphHeight;
				}
				if (IndexPos < AtlasSizeSqr && IndexPos >= 0) {
					AtlasData[IndexPos] = Character->RasterizedData[i * Character->Size.x + j];
				}
				IndexPos++;
			}
			IndexPos += -Character->Size.x + AtlasSize;
		}
	}

	return AtlasData;
}
