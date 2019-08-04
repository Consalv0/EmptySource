
#include "../include/Core.h"
#include "../include/Utility/TexturePacking.h"
#include "../include/Utility/LogFreeType.h"
#include "../include/Utility/LogCore.h"
#include "../include/Text2DGenerator.h"
#include "../include/SDFGenerator.h"
#include "../include/Utility/Timer.h"

namespace EmptySource {

	void Text2DGenerator::PrepareCharacters(const WChar * Characters, const size_t & Count) {
		TextFont->SetGlyphHeight(GlyphHeight);
		for (size_t Character = 0; Character < Count; ++Character) {
			FontGlyph Glyph;
			if (TextFont->GetGlyph(Glyph, Characters[Character])) {
				if (!Glyph.VectorShape.Validate())
					Debug::Log(Debug::LogWarning, L"The geometry of the loaded shape is invalid.");
				Glyph.VectorShape.Normalize();

				Glyph.GenerateSDF(PixelRange);
				LoadedCharacters.insert_or_assign(Characters[Character], new FontGlyph(Glyph));
			}
		}
	}

	void Text2DGenerator::PrepareCharacters(const unsigned long & From, const unsigned long & To) {
		Debug::Log(Debug::LogInfo, L"Loading %d font glyphs from %lc(%d) to %lc(%d)", To - From, From, From, To, To);
		Debug::Timer Timer;
		Timer.Start();
		TextFont->SetGlyphHeight(GlyphHeight);
		for (unsigned long Character = From; Character <= To; Character++) {
			FontGlyph Glyph;
			if (TextFont->GetGlyph(Glyph, (unsigned int)Character)) {
				if (!Glyph.VectorShape.Validate())
					Debug::Log(Debug::LogInfo | Debug::LogWarning, L"├> The geometry of the loaded shape is invalid.");
				Glyph.VectorShape.Normalize();

				Glyph.GenerateSDF(PixelRange);
				LoadedCharacters.insert_or_assign(Character, new FontGlyph(Glyph));
			}
		}
		Timer.Stop();
		Debug::Log(Debug::LogInfo, L"└> Glyphs loaded in %.3fs", Timer.GetEnlapsedSeconds());
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

			FontGlyph * Glyph = LoadedCharacters[*Character];
			if (Glyph == NULL) {
				CursorPivot.x += (PixelRange * 0.5F + GlyphHeight * 0.5F) * ScaleFactor;
				continue;
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
			FontGlyph * Glyph = LoadedCharacters[*Character];
			if (Glyph == NULL) {
				Pivot.x += (PixelRange * 0.5F + GlyphHeight * 0.5F) * ScaleFactor;
				continue;
			}

			Pivot.x += (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
		}

		return Pivot;
	}

	int Text2DGenerator::FindCharacters(const WString & InText) {
		TextFont->SetGlyphHeight(GlyphHeight);
		int Count = 0;
		for (WString::const_iterator Character = InText.begin(); Character != InText.end(); Character++) {
			if (LoadedCharacters.find(*Character) == LoadedCharacters.end()) {
				Count++; FontGlyph Glyph;
				if (TextFont->GetGlyph(Glyph, (unsigned int)*Character)) {
					if (!Glyph.VectorShape.Validate())
						Debug::Log(Debug::LogInfo | Debug::LogWarning, L" The geometry of the loaded shape is invalid.");
					Glyph.VectorShape.Normalize();

					Glyph.GenerateSDF(PixelRange);
					LoadedCharacters.insert_or_assign(*Character, new FontGlyph(Glyph));
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

		for (int i = 0; i < AtlasSizeSqr; i++) {
			// if (i % GlyphHeight == 0 || (i / (AtlasSize)) % GlyphHeight == 0) {
			// 	Atlas[i] = 255;
			// }
			// else {
			Atlas[i].R = 0;
			// }
		}

		TexturePacking<Bitmap<FloatRed>> TextureAtlas;
		TextureAtlas.CreateTexture({ AtlasSize, AtlasSize });

		size_t Count = 0;
		TArray<FontGlyph *> GlyphArray = TArray<FontGlyph * >(LoadedCharacters.size());
		for (TDictionary<unsigned long, FontGlyph*>::const_iterator Begin = LoadedCharacters.begin(); Begin != LoadedCharacters.end(); Begin++)
			GlyphArray[Count++] = Begin->second;
		std::sort(GlyphArray.begin(), GlyphArray.end(), [](FontGlyph * A, FontGlyph * B) {
			return Math::Max(A->Width, A->Height) / Math::Min(A->Width, A->Height) * A->Width * A->Height
		> Math::Max(B->Width, B->Height) / Math::Min(B->Width, B->Height) * B->Width * B->Height;
		});

		for (TArray<FontGlyph * >::const_iterator Begin = GlyphArray.begin(); Begin != GlyphArray.end(); Begin++) {
			FontGlyph * Character = *Begin;
			TexturePacking<Bitmap<FloatRed>>::ReturnElement ResultNode = TextureAtlas.Insert(Character->SDFResterized);
			if (!ResultNode.bValid || ResultNode.Element == NULL) {
				Debug::Log(Debug::LogError, L"Error writting texture of %lc(%d)", Character->UnicodeValue, Character->UnicodeValue);
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

}