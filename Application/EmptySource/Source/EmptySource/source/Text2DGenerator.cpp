
#include "..\include\Core.h"
#include "..\include\Utility\LogFreeType.h"
#include "..\include\Utility\LogCore.h"
#include "..\include\Text2DGenerator.h"
#include "..\include\Utility\Timer.h"

#include "..\include\SDFTextureGenerator.h"

void Text2DGenerator::PrepareCharacters(const unsigned long & From, const unsigned long & To) {
	TextFont->SetGlyphHeight(GlyphHeight);
	for (unsigned long Character = From; Character <= To; Character++) {
		FontGlyph Glyph;
		if (TextFont->GetGlyph(Glyph, Character)) {
			LoadedCharacters.insert_or_assign(Character, new FontGlyph(Glyph));
		}
	}
}

void Text2DGenerator::GenerateMesh(Vector2 Pivot, float HeightSize, const WString & InText, MeshFaces * Faces, MeshVertices * Vertices) {
	
	if (InText.size() == 0) return;

	float ScaleFactor = HeightSize / GlyphHeight;
	size_t InTextSize = InText.size();
	size_t InitialFacesSize = Faces->size();
	size_t InitialVerticesSize = Vertices->size();

	Faces->resize(InitialFacesSize + (InTextSize * 2));
	Vertices->resize(InitialVerticesSize + (InTextSize * 4));
	
	int VertexCount = (int)InitialVerticesSize;
	IntVector3 * TextFacesEnd = &Faces->at(InitialFacesSize);
	MeshVertex * TextVerticesEnd = &Vertices->at(InitialVerticesSize);

	// --- Iterate through all characters
	for (WString::const_iterator Character = InText.begin(); Character != InText.end(); Character++) {
		FontGlyph * Glyph = LoadedCharacters[*Character];
		if (Glyph == NULL) {
			Pivot.x += GlyphHeight * 0.5F * ScaleFactor;
			continue;
		}

		Glyph->GetQuadMesh(Pivot, ScaleFactor, TextVerticesEnd);
		TextFacesEnd->x = VertexCount;
		TextFacesEnd->y = VertexCount + 1;
		(TextFacesEnd++)->z = VertexCount + 2;
		TextFacesEnd->x = VertexCount + 3;
		TextFacesEnd->y = VertexCount;
		(TextFacesEnd++)->z = VertexCount + 1;

		VertexCount += 4;
		TextVerticesEnd += 4;

		// --- Advance cursor for next glyph (note that advance is number of 1/32 pixels)
		Pivot.x += (Glyph->Advance >> 6) * ScaleFactor;
	}

	// --- The VertexCount was initialized with the initial VertexCount
	VertexCount -= (int)InitialVerticesSize;
	Faces->resize(InitialFacesSize + ((VertexCount) / 2));
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
	for (TDictionary<unsigned long, FontGlyph*>::iterator Begin = LoadedCharacters.begin(); Begin != LoadedCharacters.end(); Begin++) {
		FontGlyph * Character = Begin->second;
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

	// GenerateSDFFromUChar(AtlasData, { AtlasSize , AtlasSize });
	SDFTextureGenerator::Generate(AtlasData, { AtlasSize , AtlasSize }, 2, 4, 0);
	return AtlasData;
}
