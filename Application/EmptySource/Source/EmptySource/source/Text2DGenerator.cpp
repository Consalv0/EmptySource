
#include "..\include\Core.h"
#include "..\include\Utility\LogFreeType.h"
#include "..\include\Utility\LogCore.h"
#include "..\include\Text2DGenerator.h"
#include "..\include\Utility\Timer.h"
#include "..\include\SDFGenerator.h"

void Text2DGenerator::PrepareCharacters(const unsigned long & From, const unsigned long & To) {
	TextFont->SetGlyphHeight(GlyphHeight);
	for (unsigned long Character = From; Character <= To; Character++) {
		FontGlyph Glyph;
		if (TextFont->GetGlyph(Glyph, Character)) {

			if (!Glyph.VectorShape.Validate())
				Debug::Log(Debug::LogWarning, L"The geometry of the loaded shape is invalid.");
			Glyph.VectorShape.Normalize();

			Box2D Bounds = { 
				MathConstants::Big_Number, MathConstants::Big_Number,
				-MathConstants::Big_Number, -MathConstants::Big_Number
			};

			Glyph.VectorShape.Bounds(Bounds);
			
			Glyph.GenerateSDF(GlyphHeight);
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

		Pivot.x += (Glyph->Advance) * ScaleFactor;
	}

	// --- The VertexCount was initialized with the initial VertexCount
	VertexCount -= (int)InitialVerticesSize;
	Faces->resize(InitialFacesSize + ((VertexCount) / 2));
	Vertices->resize(InitialVerticesSize + VertexCount);
}

void Text2DGenerator::Clear() {
	LoadedCharacters.clear();
}

bool Text2DGenerator::GenerateTextureAtlas(Bitmap<unsigned char> & Atlas) {
	int Value = 0;
	int AtlasSizeSqr = AtlasSize * AtlasSize;
	Atlas = Bitmap<unsigned char>(AtlasSize, AtlasSize);

	for (int i = 0; i < AtlasSizeSqr; i++) {
		// if (i % GlyphHeight == 0 || (i / (AtlasSize)) % GlyphHeight == 0) {
		// 	AtlasData[i] = 255;
		// }
		// else {
		Atlas[i] = 0;
		// }
	}

	IntVector2 AtlasPosition;
	for (TDictionary<unsigned long, FontGlyph*>::iterator Begin = LoadedCharacters.begin(); Begin != LoadedCharacters.end(); Begin++) {
		FontGlyph * Character = Begin->second;
		int IndexPos = AtlasPosition.x + AtlasPosition.y * AtlasSize;
		IndexPos += Character->Bearing.x;

		// --- Asign the current UV Position
		Character->UV.MinX = (Character->Bearing.u + AtlasPosition.u) / (float)AtlasSize;
		Character->UV.MaxX = (Character->Bearing.u + AtlasPosition.u + Character->SDFResterized.GetWidth()) / (float)AtlasSize;
		Character->UV.MinY = (AtlasPosition.v) / (float)AtlasSize;
		Character->UV.MaxY = (AtlasPosition.v + Character->SDFResterized.GetHeight()) / (float)AtlasSize;
		
		// --- If current position exceds the canvas with the next character, reset the position in Y
		AtlasPosition.y += GlyphHeight;
		if (AtlasPosition.y * AtlasSize > (AtlasSizeSqr - GlyphHeight * AtlasSize)) {
			AtlasPosition.x += GlyphHeight;
			AtlasPosition.y = 0;
		}
		
		// --- Render current character in the current position
		for (int i = 0; i < Character->SDFResterized.GetHeight(); ++i) {
			for (int j = 0; j < Character->SDFResterized.GetWidth(); ++j) {
				if (IndexPos >= AtlasSizeSqr) {
					IndexPos -= AtlasSizeSqr / GlyphHeight;
				}
				if (IndexPos < AtlasSizeSqr && IndexPos >= 0) {
					Value = Math::Clamp(int(Character->SDFResterized[i * Character->SDFResterized.GetWidth() + j] * 0x100), 0xff);
					Atlas[IndexPos] = Value;
				}
				IndexPos++;
			}
			IndexPos += -Character->SDFResterized.GetWidth() + AtlasSize;
		}
	}

	std::function<float(unsigned char)> FunctionAB = [](unsigned char Value) { return Value / 255.F; };
	std::function<unsigned char(float)> FunctionBA = [](float Value) { return (unsigned char)(Value * 255.F); };
	Bitmap<float> AtlasF;
	Atlas.ChangeType(AtlasF, FunctionAB);
	SDFGenerator::FromBitmap(AtlasF, 2, 4);
	AtlasF.ChangeType(Atlas, FunctionBA);
	return true;
}
