
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
				LoadedCharacters[Character]->Data = new unsigned char[Glyph->Size.x * Glyph->Size.y];
				memcpy(Glyph->Data, TextFont->FreeTypeFace->glyph->bitmap.buffer, Glyph->Size.x * Glyph->Size.y);
			}
		}
	}
}

void Text2DGenerator::GenerateTextMesh(const WString & InText, MeshFaces * Faces, MeshVertices * Vertices) {
	WString::const_iterator Character = InText.begin();
	Vector3 TextPosition = Vector3(0.F, (float)GlyphHeight);

	// --- Iterate through all characters
	for (; Character != InText.end(); Character++) {
		TextGlyph * Glyph = LoadedCharacters[*Character];
		if (Glyph == NULL) {
			TextPosition.x += GlyphHeight / 2;
			continue;
		}

		float XPos = TextPosition.x + Glyph->Bearing.x;
		float YPos = TextPosition.y - (Glyph->Size.y - Glyph->Bearing.y);

		float XPosWidth = XPos + (float)Glyph->Size.x;
		float YPosHeight = YPos + (float)Glyph->Size.y;

		int VertexCount = (int)Vertices->size();

		Faces->push_back({ VertexCount, VertexCount + 1, VertexCount + 2 });
		Faces->push_back({ VertexCount + 3, VertexCount, VertexCount + 1 });

		Vertices->push_back({ Vector3(XPosWidth, YPos),       0, Glyph->MinUV + Vector2(Glyph->MaxUV.u, 0.F) });
		Vertices->push_back({ Vector3(XPos, YPosHeight),      0, Glyph->MinUV + Vector2(0.F, Glyph->MaxUV.v) });
		Vertices->push_back({ Vector3(XPosWidth, YPosHeight), 0, Glyph->MinUV + Glyph->MaxUV                 });
		Vertices->push_back({ Vector3(XPos, YPos),            0, Glyph->MinUV                                });

		// --- Now advance cursor for next glyph (note that advance is number of 1/32 pixels)
		TextPosition.x += (Glyph->Advance >> 6);
	}
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
		Character->MinUV = (Vector2((float)Character->Bearing.x, 0.F) + AtlasPosition.FloatVector2()) / (float)AtlasSize;
		Character->MaxUV = Character->Size.FloatVector2() / (float)AtlasSize;
		
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
					AtlasData[IndexPos] = Character->Data[i * Character->Size.x + j];
				}
				IndexPos++;
			}
			IndexPos += -Character->Size.x + AtlasSize;
		}
	}

	return AtlasData;
}
