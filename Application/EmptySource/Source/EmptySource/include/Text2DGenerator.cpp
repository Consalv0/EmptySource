
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
		if (Error = FT_Load_Glyph(TextFont->FreeTypeFace, GlyphIndex, FT_LOAD_RENDER)) {
			Debug::Log(Debug::LogError, L"Failed to load Glyph '%c', %s", Character, FT_ErrorMessage(Error));
		}
		else {
			LoadedCharacters.insert_or_assign(Character, new TextGlyph(
				Character,
				{ (int)TextFont->FreeTypeFace->glyph->bitmap.width, (int)TextFont->FreeTypeFace->glyph->bitmap.rows },
				{ (int)TextFont->FreeTypeFace->glyph->bitmap_left, (int)TextFont->FreeTypeFace->glyph->bitmap_top },
				(int)TextFont->FreeTypeFace->glyph->advance.x
			));
			{
				TextGlyph * Glyph = LoadedCharacters[Character];
				LoadedCharacters[Character]->RasterizedData = new unsigned char[Glyph->Size.x * Glyph->Size.y];
				memcpy(Glyph->RasterizedData, TextFont->FreeTypeFace->glyph->bitmap.buffer, Glyph->Size.x * Glyph->Size.y);
			}
		}
	}
}

void Text2DGenerator::GenerateTextMesh(Vector2 Pivot, const WString & InText, MeshFaces * Faces, MeshVertices * Vertices) {
	WString::const_iterator Character = InText.begin();

	IntVector3 * TextFaces = new IntVector3[InText.size() * 2];
	MeshVertex * TextVertices = new MeshVertex[InText.size() * 4];
	IntVector3 * TextFacesEnd = TextFaces;
	MeshVertex * TextVerticesEnd = TextVertices;

	int VertexCount = 0;

	// --- Iterate through all characters
	for (; Character != InText.end(); Character++) {
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

	if (VertexCount > 0) {
		Faces->insert(Faces->end(), &TextFaces[0], TextFacesEnd);
		Vertices->insert(Vertices->end(), &TextVertices[0], TextVerticesEnd);
	}

	delete[] TextFaces;
	delete[] TextVertices;
}

void Text2DGenerator::Clear() {
	LoadedCharacters.clear();
}

unsigned char * Text2DGenerator::GenerateTextureAtlas() {
	unsigned char * AtlasData = new unsigned char[AtlasSize * AtlasSize];

	for (int i = 0; i < AtlasSize * AtlasSize; i++) {
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
		if (AtlasPosition.y * AtlasSize > (AtlasSize * AtlasSize - GlyphHeight * AtlasSize)) {
			AtlasPosition.x += GlyphHeight;
			AtlasPosition.y = 0;
		}
		
		// --- Render current character in the current position
		for (int i = Character->Size.y - 1; i >= 0; i--) {
			for (int j = 0; j < Character->Size.x; j++) {
				if (IndexPos >= AtlasSize * AtlasSize) {
					IndexPos -= AtlasSize * AtlasSize / GlyphHeight;
				}
				if (IndexPos < AtlasSize * AtlasSize && IndexPos >= 0) {
					AtlasData[IndexPos] = Character->RasterizedData[i * Character->Size.x + j];
				}
				IndexPos++;
			}
			IndexPos += -Character->Size.x + AtlasSize;
		}
	}

	return AtlasData;
}
