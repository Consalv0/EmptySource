Name: "DepthTestShader"
Parameters:
  - Uniform: _ViewMatrix
    Type: Matrix4x4
    DefaultValue: {}
  - Uniform: _MainTexture
    Type: Texture2D
    DefaultValue: "WhiteTexture"
GLSL:
  Stages:
    - StageType: Vertex
      Code: |
        ESOURCE_VERTEX_LAYOUT_INSTANCING(1,mat4,_ModelMatrix)
        ESOURCE_COMMON_VERTEX

        void main() {
          ESource_VertexCompute();
          gl_Position = Vertex.ScreenPosition;
        }

    - StageType: Pixel
      Code: |
        ESOURCE_MATRICES
        ESOURCE_VERTEX
        ESOURCE_UNIFORMS
        ESOURCE_MATERIAL

        void main() {
          vec4 Diffuse = texture(_MainTexture, Vertex.UV0);
          
          if (Diffuse.a > 0.5)
            gl_FragDepth = gl_FragCoord.z;
          else
            gl_FragDepth = 1.0;
        }
      