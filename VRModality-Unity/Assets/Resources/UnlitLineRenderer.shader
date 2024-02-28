// Useful resources for single-pass instanced rendering for AR/VR displays
// Unity Manual: https://docs.unity3d.com/Manual/SinglePassInstancing.html
// Rough setup: https://forum.unity.com/threads/stereo-geometry-shader-for-pointclouds.939954/
// VERTEX_OUTPUT_STEREO_EYE_INDEX: https://answers.unity.com/questions/1702908/geometry-shader-in-vr-stereo-rendering-mode-single.html
// Geometry shaders and single pass instanced: https://forum.unity.com/threads/how-to-do-single-pass-instanced-stereo-rendering-with-custom-geometry-shader.543049/

Shader "IVLab/UnlitLineRenderer"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _LineWidth ("Line Width", Range(0, 1)) = 0.01
        _LineNormal ("Line Normal", Vector) = (0, 0, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry+1" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma geometry geom

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                uint vertexID : SV_VertexID;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2g
            {
                float4 vertex : POSITION;
                float4 color: COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO_EYE_INDEX
            };

            struct g2f
            {
                float4 vertex : SV_POSITION;
                float4 color: COLOR;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // INPUTS
            float4 _Color;
            int _UseVisibility;
            int _PositionCount;
            float _LineWidth;
            float3 _LineNormal;
            StructuredBuffer<int> _Visibility;

            v2g vert (appdata v)
            {
                v2g o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2g, o);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                UNITY_INITIALIZE_OUTPUT_STEREO_EYE_INDEX(o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                o.vertex = v.vertex;

                o.color = 1;
                if (_UseVisibility == 1)
                {
                    if (_Visibility[v.vertexID] == 0)
                    {
                        o.color = 0.1;
                        // "Discard" at the vertex level by making this a degenerate vertex (return NaN)
                        o.vertex = asfloat(0x7fc00000);
                    }
                }
                return o;
            }

            [maxvertexcount(6)]
            void geom(line v2g i[2], inout TriangleStream<g2f> triangleStream)
            {
                g2f o;
                // Single-pass instancing setup
                UNITY_INITIALIZE_OUTPUT(g2f, o);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i[0]);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i[1]);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);


                float3 tangent = i[1].vertex - i[0].vertex;
                float3 lineOutwards = normalize(cross(tangent, _LineNormal));
 
                float4 offset = float4(lineOutwards * _LineWidth / 2.0, 0);

                // L\ triangle
                o.vertex = UnityObjectToClipPos(i[0].vertex - offset);
                triangleStream.Append(o);

                o.vertex = UnityObjectToClipPos(i[0].vertex + offset);
                triangleStream.Append(o);

                o.vertex = UnityObjectToClipPos(i[1].vertex + offset);
                triangleStream.Append(o);

                // \7 triangle
                o.vertex = UnityObjectToClipPos(i[0].vertex - offset);
                triangleStream.Append(o);

                o.vertex = UnityObjectToClipPos(i[1].vertex - offset);
                triangleStream.Append(o);

                o.vertex = UnityObjectToClipPos(i[1].vertex + offset);
                triangleStream.Append(o);

                triangleStream.RestartStrip();
            }

            fixed4 frag (g2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                // DEBUG: Color by vertex ID
                // return i.color;
                // DEBUG: Left/right eye coloring
                // return lerp(fixed4(1, 0, 0, 1), fixed4(0, 1, 0, 1), unity_StereoEyeIndex);

                fixed4 col = _Color;
                return col;
            }
            ENDCG
        }
    }
}
