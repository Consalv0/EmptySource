
#include "CoreMinimal.h"
#include "Utility/TexturePacking.h"
#include "Utility/LogFreeType.h"
#include "Fonts/Text2DGenerator.h"
#include "Fonts/SDFGenerator.h"

namespace ESource {

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

	void Text2DGenerator::PrepareCharacters(const uint32_t & From, const uint32_t & To) {
		LOG_CORE_INFO(L"Loading {0:d} font glyphs from {1:c}({2:d}) to {3:c}({4:d})", To - From + 1, (WChar)From, From, (WChar)To, To);
		Timestamp Timer;
		Timer.Begin();
		TextFont->SetGlyphHeight(GlyphHeight);
		for (unsigned long Character = From; Character <= To; Character++) {
			FontGlyph Glyph;
			if (TextFont->GetGlyph(Glyph, (uint32_t)Character)) {
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

	void Text2DGenerator::GenerateMesh(const Box2D & Box, float HeightSize, bool RightHanded, WString InText, MeshFaces * Faces, MeshVertices * Vertices) {
		if (RightHanded == true) {
			Text::ReverseByToken(InText, '\n');
		}

		if (InText.size() == 0) return;

		float ScaleFactor = HeightSize / GlyphHeight;
		size_t InTextSize = InText.size();
		size_t InitialFacesSize = Faces->size();
		size_t InitialVerticesSize = Vertices->size();

		Faces->resize(InitialFacesSize + (InTextSize * 2));
		Vertices->resize(InitialVerticesSize + (InTextSize * 4));

		int VertexCount = (int)InitialVerticesSize;
		IntVector3 * TextFacesEnd = &Faces->at(InitialFacesSize);
		StaticVertex * TextVerticesEnd = &Vertices->at(InitialVerticesSize);

		Vector2 CursorPivot = { RightHanded ? Box.MaxX : Box.MinX, Box.MaxY };

		// --- Iterate through all characters
		for (WString::const_iterator Character = InText.begin(); Character != InText.end(); Character++) {
			if (*(Character) == L'\n' || *(Character) == L'\r') {
				CursorPivot.X = RightHanded ? Box.MaxX : Box.MinX;
				CursorPivot.Y -= HeightSize;
				continue;
			}
			if (*(Character) == L'	') {
				float TabModule = fmodf(CursorPivot.X, (PixelRange * 0.5F + GlyphHeight * 0.5F) * ScaleFactor * 4);
				if (RightHanded)
					CursorPivot.X -= TabModule;
				else
					CursorPivot.X += TabModule;
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

			if (RightHanded) {
				if (CursorPivot.X - (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor < Box.MinX) {
					CursorPivot.X -= (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
					continue;
				}
			} else {
				if (CursorPivot.X + (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor > Box.MaxX) {
					CursorPivot.X += (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
					continue;
				}
			}
			if (CursorPivot.Y < Box.MinY)
				break;

			if (RightHanded)
				CursorPivot.X -= (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;

			Glyph->GetQuadMesh(CursorPivot, PixelRange, ScaleFactor, 1.F, TextVerticesEnd);
			TextFacesEnd->Z = VertexCount;
			TextFacesEnd->Y = VertexCount + 1;
			(TextFacesEnd++)->X = VertexCount + 2;
			TextFacesEnd->X = VertexCount + 3;
			TextFacesEnd->Y = VertexCount;
			(TextFacesEnd++)->Z = VertexCount + 1;

			VertexCount += 4;
			TextVerticesEnd += 4;

			if (!RightHanded)
				CursorPivot.X += (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
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
				Pivot.X += (PixelRange * 0.5F + GlyphHeight * 0.5F) * ScaleFactor;
				continue;
			}

			FontGlyph * Glyph = LoadedCharacters[*Character];
			Pivot.X += (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
		}

		return Pivot;
	}

	int Text2DGenerator::PrepareFindedCharacters(const WString & InText) {
		TextFont->SetGlyphHeight(GlyphHeight);
		int Count = 0;
		for (WString::const_iterator Character = InText.begin(); Character != InText.end(); Character++) {
			if (!IsCharacterLoaded(*Character)) {
				Count++; FontGlyph Glyph;
				if (TextFont->GetGlyph(Glyph, (uint32_t)*Character)) {
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

	bool Text2DGenerator::GenerateGlyphAtlas(PixelMap & Atlas) {
		int Value = 0;
		int AtlasSizeSqr = AtlasSize * AtlasSize;
		Atlas = PixelMap(AtlasSize, AtlasSize, 1, PF_R8);

		// ---- Clear Bitmap
		PixelMapUtility::PerPixelOperator(Atlas, [](unsigned char * Pixel, const unsigned char & Channels) { *Pixel = 0; });

		TexturePacking<PixelMap> TextureAtlas;
		TextureAtlas.CreateTexture({ AtlasSize, AtlasSize });
		size_t Count = 0;
		TArray<FontGlyph *> GlyphArray = TArray<FontGlyph * >(LoadedCharacters.size());

		for (TDictionary<unsigned long, FontGlyph*>::const_iterator Begin = LoadedCharacters.begin(); Begin != LoadedCharacters.end(); Begin++) {
			ES_CORE_ASSERT(Begin->second != NULL, "Glyph is NULL");
			GlyphArray[Count++] = Begin->second;
		}

		std::sort(GlyphArray.begin(), GlyphArray.end(), [](FontGlyph * A, FontGlyph * B) {
			return Math::Max(A->Width, A->Height) / Math::Min(A->Width, A->Height) * A->Width * A->Height > 
				   Math::Max(B->Width, B->Height) / Math::Min(B->Width, B->Height) * B->Width * B->Height;
		});

		for (TArray<FontGlyph * >::const_iterator Begin = GlyphArray.begin(); Begin != GlyphArray.end(); Begin++) {
			FontGlyph * Character = *Begin;

			if (Character->bUndefined)
				continue;

			TexturePacking<PixelMap>::ReturnElement ResultNode = TextureAtlas.Insert(Character->SDFResterized);
			if (!ResultNode.bValid || ResultNode.Element == NULL) {
				LOG_CORE_ERROR(L"Error writting texture of {0:c}({1:d})", Character->UnicodeValue, Character->UnicodeValue);
				continue;
			}

			// --- Asign the current UV Position
			Character->UV.MinX = (ResultNode.BBox.MinX) / (float)AtlasSize;
			Character->UV.MaxX = (ResultNode.BBox.MinX + Character->SDFResterized.GetWidth()) / (float)AtlasSize;
			Character->UV.MinY = (ResultNode.BBox.MinY) / (float)AtlasSize;
			Character->UV.MaxY = (ResultNode.BBox.MinY + Character->SDFResterized.GetHeight()) / (float)AtlasSize;

			int IndexPos = int(ResultNode.BBox.MinX + ResultNode.BBox.MinY * AtlasSize);

			// --- Render current character in the current position
			for (uint32_t i = 0; i < Character->SDFResterized.GetHeight(); ++i) {
				for (uint32_t j = 0; j < Character->SDFResterized.GetWidth(); ++j) {
					if (IndexPos < AtlasSizeSqr && IndexPos >= 0) {
						Value = Math::Clamp(int(*PixelMapUtility::GetFloatPixelAt(Character->SDFResterized, i * Character->SDFResterized.GetWidth() + j) * 0x100), 0xff);
						*PixelMapUtility::GetCharPixelAt(Atlas, IndexPos) = Value;
					}
					IndexPos++;
				}
				IndexPos += -(int)Character->SDFResterized.GetWidth() + AtlasSize;
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