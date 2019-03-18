
#include "..\include\Core.h"
#include "..\include\Utility\LogFreeType.h"
#include "..\include\Utility\LogCore.h"
#include "..\include\Text2DGenerator.h"
#include "..\include\Utility\Timer.h"
#include "..\include\SDFGenerator.h"

void Text2DGenerator::PrepareCharacters(const unsigned long & From, const unsigned long & To) {
	Debug::Log(Debug::LogNormal, L"Loading %d font glyphs from %c(%d) to %c(%d)", To - From, From, From, To, To);
	Debug::Timer Timer;
	Timer.Start();
	TextFont->SetGlyphHeight(GlyphHeight);
	for (unsigned long Character = From; Character <= To; Character++) {
		FontGlyph Glyph;
		if (TextFont->GetGlyph(Glyph, Character)) {
			if (!Glyph.VectorShape.Validate())
				Debug::Log(Debug::LogWarning, L"The geometry of the loaded shape is invalid.");
			Glyph.VectorShape.Normalize();
			
			Glyph.GenerateSDF(PixelRange);
			LoadedCharacters.insert_or_assign(Character, new FontGlyph(Glyph));
		}
	}
	Timer.Stop();
	Debug::Log(Debug::LogNormal, L"└> Glyphs loaded in %.3fs", Timer.GetEnlapsedSeconds());
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
			Pivot.x += (-PixelRange * 0.5F + GlyphHeight * 0.5F) * ScaleFactor;
			continue;
		}

		Glyph->GetQuadMesh(Pivot, PixelRange, ScaleFactor, TextVerticesEnd);
		TextFacesEnd->x = VertexCount;
		TextFacesEnd->y = VertexCount + 1;
		(TextFacesEnd++)->z = VertexCount + 2;
		TextFacesEnd->x = VertexCount + 3;
		TextFacesEnd->y = VertexCount;
		(TextFacesEnd++)->z = VertexCount + 1;

		VertexCount += 4;
		TextVerticesEnd += 4;

		Pivot.x += (-PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
	}

	// --- The VertexCount was initialized with the initial VertexCount
	VertexCount -= (int)InitialVerticesSize;
	Faces->resize(InitialFacesSize + ((VertexCount) / 2));
	Vertices->resize(InitialVerticesSize + VertexCount);
}

void Text2DGenerator::Clear() {
	LoadedCharacters.clear();
}

bool Text2DGenerator::GenerateGlyphAtlas(Bitmap<unsigned char> & Atlas) {
	int Value = 0;
	int AtlasSizeSqr = AtlasSize * AtlasSize;
	Atlas = Bitmap<unsigned char>(AtlasSize, AtlasSize);

	for (int i = 0; i < AtlasSizeSqr; i++) {
		// if (i % GlyphHeight == 0 || (i / (AtlasSize)) % GlyphHeight == 0) {
		// 	Atlas[i] = 255;
		// }
		// else {
			Atlas[i] = 0;
		// }
	}

	TArray<FontGlyph *> GlyphArray = TArray<FontGlyph * >(LoadedCharacters.size());
	// --- Sort by perimeter
	{
		size_t Count = 0;
		for (TDictionary<unsigned long, FontGlyph*>::const_iterator Begin = LoadedCharacters.begin(); Begin != LoadedCharacters.end(); Begin++)
			GlyphArray[Count++] = Begin->second;

		std::sort(GlyphArray.begin(), GlyphArray.end(), [](FontGlyph * A, FontGlyph * B) {
			return (
				A->SDFResterized.GetHeight() * 2 + A->SDFResterized.GetWidth() * 2 >
				B->SDFResterized.GetHeight() * 2 + B->SDFResterized.GetWidth() * 2
			);
		}); 
	}

	TArray<BoundingBox2D> EmptySpaces;
	EmptySpaces.push_back({0.F, 0.F, (float)AtlasSize, (float)AtlasSize});

	IntVector2 AtlasPosition;
	for (TArray<FontGlyph * >::const_iterator Begin = GlyphArray.begin(); Begin != GlyphArray.end(); Begin++) {
		FontGlyph * Character = *Begin;

		int CandidateSpaceIndex = -1;
		BoundingBox2D * BBox = NULL;
		for (TArray<BoundingBox2D>::const_iterator Back = EmptySpaces.end(); Back != EmptySpaces.begin(); Back--) {
			CandidateSpaceIndex++;
			BBox = &EmptySpaces[CandidateSpaceIndex];
			if (Character->SDFResterized.GetWidth() < (int)BBox->GetWidth() &&
				Character->SDFResterized.GetHeight() < (int)BBox->GetHeight())
				break;
		}

		if (BBox == NULL) {
			Debug::Log(Debug::LogDebug, L"Error writting in %c", Character->UnicodeValue);
			continue;
		}
		int Width = Character->SDFResterized.GetWidth();
		int Height = Character->SDFResterized.GetHeight();
		
		// --- Asign the current UV Position
		Character->UV.xMin = (BBox->xMin) / (float)AtlasSize;
		Character->UV.xMax = (BBox->xMin + Character->SDFResterized.GetWidth()) / (float)AtlasSize;
		Character->UV.yMin = (BBox->yMin) / (float)AtlasSize;
		Character->UV.yMax = (BBox->yMin + Character->SDFResterized.GetHeight()) / (float)AtlasSize;
		
		int IndexPos = int(BBox->xMin + BBox->yMin * AtlasSize);
		
		// --- Render current character in the current position
		for (int i = 0; i < Character->SDFResterized.GetHeight(); ++i) {
			for (int j = 0; j < Character->SDFResterized.GetWidth(); ++j) {
				if (IndexPos >= AtlasSizeSqr) {
					IndexPos -= AtlasSizeSqr / Character->SDFResterized.GetHeight();
					Debug::Log(Debug::LogDebug, L"Error writting in %c", Character->UnicodeValue);
				}
				if (IndexPos < AtlasSizeSqr && IndexPos >= 0) {
					Value = Math::Clamp(int(Character->SDFResterized[i * Character->SDFResterized.GetWidth() + j] * 0x100), 0xff);
					Atlas[IndexPos] = Value;
				}
				IndexPos++;
			}
			IndexPos += -Character->SDFResterized.GetWidth() + AtlasSize;
		}

		int DeltaWidth = (int)BBox->GetWidth() - Width;
		int DeltaHeight = (int)BBox->GetHeight() - Height;
		BoundingBox2D BiggerSplit, SmallerSplit;

		if (DeltaWidth < DeltaHeight) {
			BiggerSplit =  { BBox->xMin,         BBox->yMin + Height, BBox->xMax, BBox->yMax};
			SmallerSplit = { BBox->xMin + Width, BBox->yMin,          BBox->xMax, BBox->yMin + Height };
		}
		else {
			BiggerSplit =  { BBox->xMin + Width, BBox->yMin,          BBox->xMax,         BBox->yMax };
			SmallerSplit = { BBox->xMin,         BBox->yMin + Height, BBox->xMin + Width, BBox->yMax };
		}

		EmptySpaces[CandidateSpaceIndex] = EmptySpaces.back();
		EmptySpaces.pop_back();

		EmptySpaces.push_back(BiggerSplit);
		EmptySpaces.push_back(SmallerSplit);
	}

	return true;
}