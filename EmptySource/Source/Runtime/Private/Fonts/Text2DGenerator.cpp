
#include "CoreMinimal.h"
#include "Utility/TexturePacking.h"
#include "Utility/LogFreeType.h"
#include "Fonts/Text2DGenerator.h"
#include "Fonts/SDFGenerator.h"

namespace EmptySource {

	void Text2DGenerator::PrepareCharacters(const WChar * Characters, const size_t & Count) {
		TextFont->SetGlyphHeight(GlyphHeight);
		for (size_t Character = 0; Character < Count; ++Character) {
			FontGlyph Glyph;
			if (TextFont->GetGlyph(Glyph, Characters[Character])) {
				if (!Glyph.VectorShape.Validate())
					LOG_CORE_ERROR(L"The geometry of the loaded shape is invalid.");
				Glyph.VectorShape.Normalize();

				Glyph.GenerateSDF(PixelRange);
				LoadedCharacters.insert_or_assign(Characters[Character], new FontGlyph(Glyph));
			}
		}
	}

	void Text2DGenerator::PrepareCharacters(const unsigned long & From, const unsigned long & To) {
		LOG_CORE_INFO(L"Loading {0:d} font glyphs from {1:c}({2:d}) to {3:c}({4:d})", To - From, From, From, To, To);
		Timestamp Timer;
		Timer.Begin();
		TextFont->SetGlyphHeight(GlyphHeight);
		for (unsigned long Character = From; Character <= To; Character++) {
			FontGlyph Glyph;
			if (TextFont->GetGlyph(Glyph, (unsigned int)Character)) {
				if (!Glyph.VectorShape.Validate())
					LOG_CORE_ERROR(L"├> The geometry of the loaded shape is invalid.");
				Glyph.VectorShape.Normalize();

				Glyph.GenerateSDF(PixelRange);
				LoadedCharacters.insert_or_assign(Character, new FontGlyph(Glyph));
			}
		}
		Timer.Stop();
		LOG_CORE_INFO(L"└> Glyphs loaded in {:.3f}s", Timer.GetDeltaTime<Time::Second>());
	}

	void Text2DGenerator::GenerateMesh(const Box2D & Box, float HeightSize, const WString & InText, MeshFaces * Faces, MeshVertices * Vertices) {

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

		Vector2 CursorPivot = { Box.xMin, Box.yMax };

		// --- Iterate through all characters
		for (WString::const_iterator Character = InText.begin(); Character != InText.end(); Character++) {
			if (*(Character) == L'\n' || *(Character) == L'\r') {
				CursorPivot.x = Box.xMin;
				CursorPivot.y -= HeightSize;
				continue;
			}
			if (*(Character) == L'	') {
				float TabModule = fmodf(CursorPivot.x, (PixelRange * 0.5F + GlyphHeight * 0.5F) * ScaleFactor * 4);
				CursorPivot.x += TabModule;
				continue;
			}

			FontGlyph * Glyph = NULL;
			if (!IsCharacterLoaded(*Character)) {
				Glyph = LoadedCharacters[L'\0'];
			} else {
				Glyph = LoadedCharacters[*Character];
				if (Glyph->bUndefined) {
					Glyph = LoadedCharacters[L'\0'];
				}
			}

			if (CursorPivot.x + (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor > Box.xMax) {
				CursorPivot.x += (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
				continue;
			}
			if (CursorPivot.y < Box.yMin)
				break;

			Glyph->GetQuadMesh(CursorPivot, PixelRange, ScaleFactor, TextVerticesEnd);
			TextFacesEnd->x = VertexCount;
			TextFacesEnd->y = VertexCount + 1;
			(TextFacesEnd++)->z = VertexCount + 2;
			TextFacesEnd->x = VertexCount + 3;
			TextFacesEnd->y = VertexCount;
			(TextFacesEnd++)->z = VertexCount + 1;

			VertexCount += 4;
			TextVerticesEnd += 4;

			CursorPivot.x += (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
		}

		// --- The VertexCount was initialized with the initial VertexCount
		VertexCount -= (int)InitialVerticesSize;
		Faces->resize(InitialFacesSize + ((VertexCount) / 2));
		Vertices->resize(InitialVerticesSize + VertexCount);
	}

	Vector2 Text2DGenerator::GetLenght(float HeightSize, const WString & InText) {
		Vector2 Pivot = { 0, HeightSize / GlyphHeight };
		if (InText.size() == 0) return Pivot;

		float ScaleFactor = HeightSize / GlyphHeight;

		// --- Iterate through all characters
		for (WString::const_iterator Character = InText.begin(); Character != InText.end(); Character++) {
			if (!IsCharacterLoaded(*Character)) {
				Pivot.x += (PixelRange * 0.5F + GlyphHeight * 0.5F) * ScaleFactor;
				continue;
			}

			FontGlyph * Glyph = LoadedCharacters[*Character];
			Pivot.x += (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
		}

		return Pivot;
	}

	int Text2DGenerator::PrepareFindedCharacters(const WString & InText) {
		TextFont->SetGlyphHeight(GlyphHeight);
		int Count = 0;
		for (WString::const_iterator Character = InText.begin(); Character != InText.end(); Character++) {
			if (!IsCharacterLoaded(*Character)) {
				Count++; FontGlyph Glyph;
				if (TextFont->GetGlyph(Glyph, (unsigned int)*Character)) {
					if (!Glyph.VectorShape.Validate())
						LOG_CORE_ERROR(L"The geometry of the loaded shape is invalid.");
					Glyph.VectorShape.Normalize();

					Glyph.GenerateSDF(PixelRange);
					AddNewGlyph(Glyph);
				}
			}
		}
		return Count;
	}

	void Text2DGenerator::Clear() {
		LoadedCharacters.clear();
	}

	bool Text2DGenerator::GenerateGlyphAtlas(Bitmap<UCharRed> & Atlas) {
		int Value = 0;
		int AtlasSizeSqr = AtlasSize * AtlasSize;
		Atlas = Bitmap<UCharRed>(AtlasSize, AtlasSize);

		// ---- Clear Bitmap
		Atlas.PerPixelOperator([](EmptySource::UCharRed & Pixel) { Pixel.R = 0; });

		TexturePacking<Bitmap<FloatRed>> TextureAtlas;
		TextureAtlas.CreateTexture({ AtlasSize, AtlasSize });
		size_t Count = 0;
		TArray<FontGlyph *> GlyphArray = TArray<FontGlyph * >(LoadedCharacters.size());

		for (TDictionary<unsigned long, FontGlyph*>::const_iterator Begin = LoadedCharacters.begin(); Begin != LoadedCharacters.end(); Begin++)
			GlyphArray[Count++] = Begin->second;

		std::sort(GlyphArray.begin(), GlyphArray.end(), [](FontGlyph * A, FontGlyph * B) {
			ES_CORE_ASSERT(A, "Glyph is NULL");
			ES_CORE_ASSERT(B, "Glyph is NULL");
			return Math::Max(A->Width, A->Height) / Math::Min(A->Width, A->Height) * A->Width * A->Height > 
				   Math::Max(B->Width, B->Height) / Math::Min(B->Width, B->Height) * B->Width * B->Height;
		});

		for (TArray<FontGlyph * >::const_iterator Begin = GlyphArray.begin(); Begin != GlyphArray.end(); Begin++) {
			FontGlyph * Character = *Begin;
			ES_CORE_ASSERT(Character != NULL, "Glyph is NULL");

			if (Character->bUndefined)
				continue;

			TexturePacking<Bitmap<FloatRed>>::ReturnElement ResultNode = TextureAtlas.Insert(Character->SDFResterized);
			if (!ResultNode.bValid || ResultNode.Element == NULL) {
				LOG_CORE_ERROR(L"Error writting texture of {0:c}({1:d})", Character->UnicodeValue, Character->UnicodeValue);
				continue;
			}

			// --- Asign the current UV Position
			Character->UV.xMin = (ResultNode.BBox.xMin) / (float)AtlasSize;
			Character->UV.xMax = (ResultNode.BBox.xMin + Character->SDFResterized.GetWidth()) / (float)AtlasSize;
			Character->UV.yMin = (ResultNode.BBox.yMin) / (float)AtlasSize;
			Character->UV.yMax = (ResultNode.BBox.yMin + Character->SDFResterized.GetHeight()) / (float)AtlasSize;

			int IndexPos = int(ResultNode.BBox.xMin + ResultNode.BBox.yMin * AtlasSize);

			// --- Render current character in the current position
			for (int i = 0; i < Character->SDFResterized.GetHeight(); ++i) {
				for (int j = 0; j < Character->SDFResterized.GetWidth(); ++j) {
					if (IndexPos < AtlasSizeSqr && IndexPos >= 0) {
						Value = Math::Clamp(int(Character->SDFResterized[i * Character->SDFResterized.GetWidth() + j].R * 0x100), 0xff);
						Atlas[IndexPos].R = Value;
					}
					IndexPos++;
				}
				IndexPos += -Character->SDFResterized.GetWidth() + AtlasSize;
			}
		}

		return true;
	}

	bool Text2DGenerator::IsCharacterLoaded(unsigned long Character) const {
		return LoadedCharacters.find(Character) != LoadedCharacters.end();
	}

	void Text2DGenerator::AddNewGlyph(const FontGlyph & Glyph) {
		LoadedCharacters.insert_or_assign(Glyph.UnicodeValue, new FontGlyph(Glyph));
	}

}