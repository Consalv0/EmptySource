#pragma once

#include "Rendering/Mesh.h"
#include "Math/Vector3.h"

namespace ESource {

	namespace MeshPrimitives {
		extern Mesh Cube;
		extern Mesh Quad;

		static MeshData CreateCubeMeshData(const Vector3 & Position, const Vector3 & Size) {
			MeshData Data;
			Data.Name = "Cube";
			Data.Faces = {
				{  0,  1,  2 }, {  3,  4,  5 },
				{  6,  7,  8 },	{  9, 10, 11 },
				{ 12, 13, 14 },	{ 15, 16, 17 },
				{  0, 18,  1 },	{  3, 19,  4 },
				{  6, 20,  7 },	{  9, 21, 10 },
				{ 12, 22, 13 },	{ 15, 23, 16 }
			};

			Data.MaterialsMap.emplace(0, "default");
			Data.SubdivisionsMap.emplace(0, Subdivision({ 0, 0, 0, 12 * 3 }));

			Matrix4x4 Transform = Matrix4x4::Translation(Position) * Matrix4x4::Scaling(Size);
			Data.StaticVertices = {
				{ Transform.MultiplyPoint({ 0.5F, -0.5F,  0.5F}), { 0.F, -1.F,  0.F}, { 0.F,  0.F, 1.F}, {1.F, 0.F}, {1.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F, -0.5F, -0.5F}), { 0.F, -1.F,  0.F}, { 0.F,  0.F, 1.F}, {0.F, 1.F}, {0.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F, -0.5F, -0.5F}), { 0.F, -1.F,  0.F}, {-0.F,  0.F, 1.F}, {0.F, 0.F}, {0.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F,  0.5F, -0.5F}), { 0.F,  1.F,  0.F}, {-1.F,  0.F, 0.F}, {1.F, 0.F}, {1.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F,  0.5F,  0.5F}), { 0.F,  1.F,  0.F}, {-1.F,  0.F, 0.F}, {0.F, 1.F}, {0.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F,  0.5F, -0.5F}), { 0.F,  1.F,  0.F}, {-1.F,  0.F, 0.F}, {0.F, 0.F}, {0.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F,  0.5F, -0.5F}), { 1.F, -0.F,  0.F}, { 0.F,  1.F, 0.F}, {1.F, 0.F}, {1.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F, -0.5F,  0.5F}), { 1.F, -0.F,  0.F}, { 0.F,  1.F, 0.F}, {0.F, 1.F}, {0.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F, -0.5F, -0.5F}), { 1.F, -0.F,  0.F}, { 0.F,  1.F, 0.F}, {0.F, 0.F}, {0.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F,  0.5F,  0.5F}), { 0.F, -0.F,  1.F}, { 0.F,  1.F, 0.F}, {1.F, 0.F}, {1.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F, -0.5F,  0.5F}), { 0.F, -0.F,  1.F}, { 0.F,  1.F, 0.F}, {0.F, 1.F}, {0.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F, -0.5F,  0.5F}), { 0.F, -0.F,  1.F}, {-0.F,  1.F, 0.F}, {0.F, 0.F}, {0.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F, -0.5F,  0.5F}), {-1.F, -0.F, -0.F}, { 0.F,  1.F, 0.F}, {0.F, 0.F}, {0.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F,  0.5F, -0.5F}), {-1.F, -0.F, -0.F}, { 0.F,  1.F, 0.F}, {1.F, 1.F}, {1.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F, -0.5F, -0.5F}), {-1.F, -0.F, -0.F}, { 0.F,  1.F, 0.F}, {0.F, 1.F}, {0.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F, -0.5F, -0.5F}), { 0.F,  0.F, -1.F}, { 0.F, -1.F, 0.F}, {1.F, 0.F}, {1.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F,  0.5F, -0.5F}), { 0.F,  0.F, -1.F}, { 0.F, -1.F, 0.F}, {0.F, 1.F}, {0.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F,  0.5F, -0.5F}), { 0.F,  0.F, -1.F}, {-0.F, -1.F, 0.F}, {0.F, 0.F}, {0.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F, -0.5F,  0.5F}), { 0.F, -1.F,  0.F}, { 0.F,  0.F, 1.F}, {1.F, 1.F}, {1.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F,  0.5F,  0.5F}), { 0.F,  1.F,  0.F}, {-1.F,  0.F, 0.F}, {1.F, 1.F}, {1.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({ 0.5F,  0.5F,  0.5F}), { 1.F, -0.F,  0.F}, { 0.F,  1.F, 0.F}, {1.F, 1.F}, {1.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F,  0.5F,  0.5F}), { 0.F, -0.F,  1.F}, { 0.F,  1.F, 0.F}, {1.F, 1.F}, {1.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F,  0.5F,  0.5F}), {-1.F, -0.F, -0.F}, { 0.F,  1.F, 0.F}, {1.F, 0.F}, {1.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-0.5F, -0.5F, -0.5F}), { 0.F,  0.F, -1.F}, { 0.F, -1.F, 0.F}, {1.F, 1.F}, {1.F, 1.F}, {1.F} }
			};

			Data.hasBoundingBox = false;
			Data.hasNormals = true;
			Data.hasVertexColor = true;
			Data.UVChannels = 1;
			Data.ComputeTangents();
			Data.ComputeBounding();
			return Data;
		}

		static MeshData CreateQuadMeshData(const Vector3 & Position, const Vector3 & Size) {
			MeshData Data;
			Data.Name = "Quad";
			Data.Faces = {
				{0, 1, 2}, {0, 3, 1}
			};

			Data.MaterialsMap.emplace(0, "default");
			Data.SubdivisionsMap.emplace(0, Subdivision({ 0, 0, 0, 2 * 3 }));

			Matrix4x4 Transform = Matrix4x4::Translation(Position) * Matrix4x4::Scaling(Size);
			Data.StaticVertices = {
				{ Transform.MultiplyPoint({ 1.F, -1.F, -0.F}), {0.F, 0.F, 1.F}, {1.F, -0.F, -0.F}, {1.F, 0.F}, {1.F, 0.F}, {1.F} },
				{ Transform.MultiplyPoint({-1.F,  1.F,  0.F}), {0.F, 0.F, 1.F}, {1.F, -0.F, -0.F}, {0.F, 1.F}, {0.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({ 1.F,  1.F,  0.F}), {0.F, 0.F, 1.F}, {1.F, -0.F, -0.F}, {1.F, 1.F}, {1.F, 1.F}, {1.F} },
				{ Transform.MultiplyPoint({-1.F, -1.F, -0.F}), {0.F, 0.F, 1.F}, {1.F, -0.F, -0.F}, {0.F, 0.F}, {0.F, 0.F}, {1.F} }
			};

			Data.hasBoundingBox = false;
			Data.hasNormals = true;
			Data.hasVertexColor = true;
			Data.UVChannels = 1;
			Data.ComputeTangents();
			Data.ComputeBounding();
			return Data;
		}

		static void Initialize() {
			MeshData CubeData = CreateCubeMeshData(0.F, 1.F);
			Cube.SwapMeshData(CubeData);
			Cube.SetUpBuffers();
			MeshData QuadData = CreateQuadMeshData(0.F, 1.F);
			Quad.SwapMeshData(QuadData);
			Quad.SetUpBuffers();
		}
	}

}