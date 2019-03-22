
#include "../include/Core.h"
#include "../include/Utility/LogFreeType.h"
#include "../include/Utility/LogCore.h"
#include "../include/Text2DGenerator.h"
#include "../include/Utility/Timer.h"
#include "../include/SDFGenerator.h"

Text2DGenerator::Node * Text2DGenerator::Node::Insert(const FontGlyph & Glyph) {
    // --- We're not in leaf
	if (Smaller != NULL) {
		// --- Try inserting into first child
		Node * NewNode = Smaller->Insert(Glyph);
		if (NewNode != NULL) return NewNode;

		// --- No room, insert into second
		return Bigger->Insert(Glyph);
	}
    else {
		// --- If There's already a glyph here, return;
		if (this->Glyph != NULL) return NULL;

		// --- If We're too small, return
		if (Glyph.SDFResterized.GetWidth() > (int)BBox.GetWidth() ||
			Glyph.SDFResterized.GetHeight() > (int)BBox.GetHeight())
			return NULL;

        // --- If We're just right, accept
        if (Glyph.SDFResterized.GetWidth() == (int)BBox.GetWidth() &&
			Glyph.SDFResterized.GetWidth() == (int)BBox.GetHeight())
            return this;
        
        // --- Otherwise, gotta split this node and create some kids
		Smaller = new Node();
		Bigger  = new Node();
        
		int Width = Glyph.SDFResterized.GetWidth();
		int Height = Glyph.SDFResterized.GetHeight();

		int DeltaWidth = (int)BBox.GetWidth() - Width;
		int DeltaHeight = (int)BBox.GetHeight() - Height;

        // --- Decide which way to split
		if (DeltaWidth < DeltaHeight) {
			Smaller->BBox = { BBox.xMin + Width, BBox.yMin, BBox.xMax, BBox.yMin + Height};
			Bigger->BBox  = { BBox.xMin, BBox.yMin + Height, BBox.xMax, BBox.yMax };
		}
        else {
            Smaller->BBox = { BBox.xMin, BBox.yMin + Height, BBox.xMin + Width, BBox.yMax };
			Bigger->BBox  = { BBox.xMin + Width, BBox.yMin, BBox.xMax, BBox.yMax };
        }

		this->Glyph = &Glyph;
		return this;
	}
}

Text2DGenerator::Node::~Node() {
	if (Smaller != NULL)
		delete Smaller;
	if (Bigger != NULL)
		delete Bigger;
}

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
	Debug::Log(Debug::LogNormal, L"Loading %d font glyphs from %lc(%d) to %lc(%d)", To - From, From, From, To, To);
	Debug::Timer Timer;
	Timer.Start();
	TextFont->SetGlyphHeight(GlyphHeight);
	for (unsigned long Character = From; Character <= To; Character++) {
		FontGlyph Glyph;
		if (TextFont->GetGlyph(Glyph, (unsigned int)Character)) {
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
			Pivot.x += (PixelRange * 0.5F + GlyphHeight * 0.5F) * ScaleFactor;
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

		Pivot.x += (PixelRange * 0.5F + Glyph->Advance) * ScaleFactor;
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
	
	Node PNode = Node();
	PNode.BBox = {0.F, 0.F, (float)AtlasSize, (float)AtlasSize};

	size_t Count = 0;
	TArray<FontGlyph *> GlyphArray = TArray<FontGlyph * >(LoadedCharacters.size());
	for (TDictionary<unsigned long, FontGlyph*>::const_iterator Begin = LoadedCharacters.begin(); Begin != LoadedCharacters.end(); Begin++)
		GlyphArray[Count++] = Begin->second;

	std::sort(GlyphArray.begin(), GlyphArray.end(), [](FontGlyph * A, FontGlyph * B) {
		return A->SDFResterized.GetHeight() * A->SDFResterized.GetWidth() 
			 > B->SDFResterized.GetHeight() * B->SDFResterized.GetWidth();
	});

	for (TArray<FontGlyph * >::const_iterator Begin = GlyphArray.begin(); Begin != GlyphArray.end(); Begin++) {
		FontGlyph * Character = *Begin;
		Node * ResultNode = PNode.Insert(*Character);
		if (ResultNode == NULL || ResultNode->Glyph == NULL) {
			Debug::Log(Debug::LogError, L"Error writting in %lc(%d)", Character->UnicodeValue, Character->UnicodeValue);
			continue;
		}

		// --- Asign the current UV Position
		Character->UV.xMin = (ResultNode->BBox.xMin) / (float)AtlasSize;
		Character->UV.xMax = (ResultNode->BBox.xMin + Character->SDFResterized.GetWidth()) / (float)AtlasSize;
		Character->UV.yMin = (ResultNode->BBox.yMin) / (float)AtlasSize;
		Character->UV.yMax = (ResultNode->BBox.yMin + Character->SDFResterized.GetHeight()) / (float)AtlasSize;

		int IndexPos = int(ResultNode->BBox.xMin + ResultNode->BBox.yMin * AtlasSize);

		// --- Render current character in the current position
		for (int i = 0; i < Character->SDFResterized.GetHeight(); ++i) {
			for (int j = 0; j < Character->SDFResterized.GetWidth(); ++j) {
				if (IndexPos < AtlasSizeSqr && IndexPos >= 0) {
					Value = Math::Clamp(int(Character->SDFResterized[i * Character->SDFResterized.GetWidth() + j] * 0x100), 0xff);
					Atlas[IndexPos] = Value;
				}
				IndexPos++;
			}
			IndexPos += -Character->SDFResterized.GetWidth() + AtlasSize;
		}
	}

	return true;
}
