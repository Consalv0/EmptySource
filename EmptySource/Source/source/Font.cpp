
#include "../External/FreeType/include/ft2build.h"
#include FT_FREETYPE_H
#include FT_OUTLINE_H
#include "../External/FreeType/include/freetype/freetype.h"

#include "../include/Core.h"
#include "../include/Utility/LogFreeType.h"
#include "../include/Utility/LogCore.h"
#include "../include/FileStream.h"
#include "../include/Font.h"

namespace EmptySource {

	FT_LibraryRec_ * Font::FreeTypeLibrary = NULL;

	bool Font::InitializeFreeType() {
		// --- Free Type 2 Library is already instanciated
		if (FreeTypeLibrary != NULL) {
			return true;
		}

		FT_Error Error;
		if ((Error = FT_Init_FreeType(&FreeTypeLibrary))) {
			Debug::Log(Debug::LogCritical, L"Could not initialize FreeType Library, %ls", FT_ErrorMessage(Error));
			return false;
		}

		return true;
	}

	Font::Font() {
		Face = NULL;
	}

	void Font::Clear() {
		FT_Done_Face(Face);
	}

	void Font::SetGlyphHeight(const unsigned int & Size) const {
		FT_Error Error = FT_Set_Pixel_Sizes(Face, 0, Size);
		if (Error) {
			Debug::Log(Debug::LogError, L"Couldn't set the glyph size to %d, %ls", Size, FT_ErrorMessage(Error));
		}
	}

	Point2 FT_Point2(const FT_Vector & Vector) {
		return Point2(Vector.x / 64.F, Vector.y / 64.F);
	}

	int FT_MoveTo(const FT_Vector * To, void * User) {
		FT_Context *Context = reinterpret_cast<FT_Context *>(User);
		Context->contour = &Context->shape->AddContour();
		Context->position = FT_Point2(*To);
		return 0;
	}

	int FT_LineTo(const FT_Vector * To, void * User) {
		FT_Context *Context = reinterpret_cast<FT_Context *>(User);
		Context->contour->AddEdge(new LinearSegment(Context->position, FT_Point2(*To)));
		Context->position = FT_Point2(*To);
		return 0;
	}

	int FT_ConicTo(const FT_Vector * Control, const FT_Vector * To, void * User) {
		FT_Context *Context = reinterpret_cast<FT_Context *>(User);
		Context->contour->AddEdge(new QuadraticSegment(Context->position, FT_Point2(*Control), FT_Point2(*To)));
		Context->position = FT_Point2(*To);
		return 0;
	}

	int FT_CubicTo(const FT_Vector * ControlA, const FT_Vector * ControlB, const FT_Vector * To, void * User) {
		FT_Context *Context = reinterpret_cast<FT_Context *>(User);
		Context->contour->AddEdge(new CubicSegment(Context->position, FT_Point2(*ControlA), FT_Point2(*ControlB), FT_Point2(*To)));
		Context->position = FT_Point2(*To);
		return 0;
	}

	int FT_Shift(const FT_Vector * ControlA, const FT_Vector * ControlB, const FT_Vector * To, void * User) {
		FT_Context *Context = reinterpret_cast<FT_Context *>(User);
		Context->contour->AddEdge(new CubicSegment(Context->position, FT_Point2(*ControlA), FT_Point2(*ControlB), FT_Point2(*To)));
		Context->position = FT_Point2(*To);
		return 0;
	}

	void Font::Initialize(FileStream * File) {
		FT_Error Error = 0;
		if (File == NULL || (Error = FT_New_Face(FreeTypeLibrary, WStringToString(File->GetPath()).c_str(), 0, &Face)))
			Debug::Log(Debug::LogError, L"Failed to load font, %ls", FT_ErrorMessage(Error));
	}

	unsigned int Font::GetGlyphIndex(const unsigned long & Character) const {
		return FT_Get_Char_Index(Face, Character);
	}

	bool Font::GetGlyph(FontGlyph & Glyph, const unsigned int& Character) {
		FT_Error Error = FT_Load_Glyph(Face, GetGlyphIndex(Character), FT_LOAD_COMPUTE_METRICS);
		if (Error) {
			Debug::Log(Debug::LogError, L"Failed to load Glyph '%lc', %ls", Character, FT_ErrorMessage(Error));
			return false;
		}
		FT_GlyphSlot & FTGlyph = Face->glyph;

		Glyph.UnicodeValue = Character;
		Glyph.VectorShape.Contours.clear();
		Glyph.VectorShape.bInverseYAxis = false;
		Glyph.Advance = FTGlyph->metrics.horiAdvance / 64.F;
		Glyph.Width = FTGlyph->metrics.width / 64.F;
		Glyph.Height = FTGlyph->metrics.height / 64.F;
		Glyph.Bearing.x = (int)FTGlyph->metrics.horiBearingX / 64;
		Glyph.Bearing.y = (int)FTGlyph->metrics.horiBearingY / 64;

		FT_Context Context = { };
		Context.shape = &Glyph.VectorShape;
		FT_Outline_Funcs Functions;
		Functions.move_to = &FT_MoveTo;
		Functions.line_to = &FT_LineTo;
		Functions.conic_to = &FT_ConicTo;
		Functions.cubic_to = &FT_CubicTo;
		Functions.shift = 0;
		Functions.delta = 0;
		Error = FT_Outline_Decompose(&FTGlyph->outline, &Functions, &Context);
		if (Error) {
			Debug::Log(Debug::LogError, L"Failed to decompose outline of Glyph '%lc', %ls", Character, FT_ErrorMessage(Error));
			return false;
		}
		return true;
	}

}