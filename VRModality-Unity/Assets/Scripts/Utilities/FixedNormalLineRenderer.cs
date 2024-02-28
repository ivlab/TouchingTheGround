using System;
using System.Linq;
using UnityEngine;
using System.Collections.Generic;

namespace IVLab.Utilities
{
    /// <summary>
    /// Replacement LineRenderer with a minimal set of features for drawing line
    /// along a single dimension. Builds a 3D mesh out of a series of points.
    /// </summary>
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class FixedNormalLineRenderer : MonoBehaviour
    {
        public Material material;
        public int positionCount;
        public float width;
        public Vector3 lineNormal = Vector3.forward;

        private Vector3[] positions;
        private MeshFilter mf;
        private MeshRenderer mr;
        private bool[] visibility;
        private ComputeBuffer visibilityBuffer;
        private bool positionsUpdated = false;
        private bool visibilityUpdated = false;
        private bool indicesUpdated = false;
        private Mesh mesh;
        private Vector3[] meshVertices;
        private int[] meshTriangleIndices;
        private HashSet<int> updatedPositionIndices;
        private int[] visibilityInts;

        void Awake()
        {
            updatedPositionIndices = new HashSet<int>();
            if (material == null)
            {
                Shader unlit = Shader.Find("IVLab/UnlitLineRenderer");
                material = new Material(unlit);
            }
        }

        void OnDestroy()
        {
            visibilityBuffer?.Dispose();
        }

        /// <summary>
        /// Set the vertex positions of the line
        /// </summary>
        public void SetPositions(Vector3[] positions)
        {
            if (positionCount < 2)
            {
                Debug.LogError("FixedNormalLineRenderer: Line must have at least 2 vertices.");
            }

            this.positions = positions;
            positionsUpdated = true;
        }

        /// <summary>
        /// Set (or override) a single position in the line
        /// </summary>
        public void SetPosition(Vector3 position, int index)
        {
            if (positionCount < 2)
            {
                Debug.LogError("FixedNormalLineRenderer: Line must have at least 2 vertices.");
            }

            if (index < 0 || index > positionCount)
            {
                Debug.LogError("FixedNormalLineRenderer: Index out of range when setting positions");
            }

            this.positions[index] = position;
            updatedPositionIndices.Add(index);
            positionsUpdated = true;
        }

        /// <summary>
        /// Set per-index visibility of this line
        /// </summary>
        public void SetVisibility(bool[] visibility)
        {
            this.visibility = visibility;
            visibilityUpdated = true;
        }

        /// <summary>
        /// Set per-index visibility of this line (for a single vertex)
        /// </summary>
        public void SetVisibility(bool visible, int index)
        {
            if (index < 0 || index > positionCount)
            {
                Debug.LogError("FixedNormalLineRenderer: Index out of range when setting visibility");
            }
            this.visibility[index] = visible;
            visibilityUpdated = true;
        }

        public Vector3 GetPosition(int index)
        {
            return this.positions[index];
        }

        public Vector3[] GetPositions()
        {
            return this.positions;
        }

        /// <summary>
        /// Build a 3D line out of the given line geometry.
        /// </summary>
        public void RecalculateGeometry()
        {
            if (mf == null)
                mf = GetComponent<MeshFilter>();
            if (mr == null)
                mr = GetComponent<MeshRenderer>();
            // Only recalculate full geometry if necessary
            if (positionsUpdated)
            {
                if (mesh == null)
                {
                    mesh = new Mesh();
                    mesh.name = "Fixed Normal Line Renderer " + DateTimeOffset.Now.ToString();
                }

                // Indexing: Line Strip. (0 1 2 3, ...) defines a line going through all those indices.
                if (meshVertices == null || meshVertices.Length != positionCount * 1)
                {
                    meshVertices = new Vector3[positionCount * 1];
                }
                if (meshTriangleIndices == null || meshTriangleIndices.Length != positionCount * 1)
                {
                    // meshTriangleIndices = new int[positionCount * 1];
                    meshTriangleIndices = Enumerable.Range(0, positionCount).ToArray();
                    indicesUpdated = true;
                }

                // Add the original positions
                Array.Copy(positions, meshVertices, positionCount);

                // Add the offset vertices
                if (updatedPositionIndices.Count == 0)
                {
                    for (int i = 0; i < positionCount; i++)
                    {
                        UpdateMeshAtIndex(i);
                    }
                }
                else
                {
                    foreach (int index in updatedPositionIndices)
                    {
                        UpdateMeshAtIndex(index);
                    }
                    updatedPositionIndices.Clear();
                }

                mesh.SetVertices(meshVertices);
                if (indicesUpdated)
                {
                    mesh.SetIndices(meshTriangleIndices, MeshTopology.LineStrip, 0);
                    indicesUpdated = false;
                }

                mf.sharedMesh = mesh;
            }

            MaterialPropertyBlock block = new MaterialPropertyBlock();

            // If there's per-index visibility, use it
            if (visibility != null && visibilityUpdated)
            {
                // Set up the compute buffer
                // NOTE: This is horribly inefficient, it uses at least 32x the
                // amount of memory it needs to because it stores a bit value in
                // a 4-byte integer.
                int stride = sizeof(int);
                if (visibilityInts == null || visibilityInts.Length != positionCount * 1)
                    visibilityInts = new int[positionCount * 1];

                // Copy data over and replicate the vertex data for the 3 stacked vertical
                for (int i = 0; i < positionCount; i++)
                {
                    visibilityInts[i] = Convert.ToInt32(visibility[i]);
                }

                // Set stuff on shader
                if (visibilityBuffer == null)
                    visibilityBuffer = new ComputeBuffer(visibilityInts.Length, stride, ComputeBufferType.Default);
                visibilityBuffer.SetData(visibilityInts);
                block.SetBuffer("_Visibility", visibilityBuffer);
                block.SetInt("_UseVisibility", 1);
            }
            else
            {
                block.SetInt("_UseVisibility", 0);
            }

            // Set properties on shader
            block.SetInt("_PositionCount", positionCount);
            block.SetFloat("_LineWidth", width);
            block.SetVector("_LineNormal", lineNormal);
            mr.SetPropertyBlock(block);

            mr.sharedMaterial = material;

            positionsUpdated = false;
            visibilityUpdated = false;
        }

        private void UpdateMeshAtIndex(int index)
        {
            if (mesh == null)
                return;

            Vector3 normal = lineNormal.normalized;

            float lineT = index / (float)positionCount;
            Vector3 right = Vector3.zero;
            if (index < positionCount - 1)
            {
                right = (positions[index + 1] - positions[index]).normalized;
            }
            else
            {
                right = (positions[index] - positions[index - 1]).normalized;
            }

            if (index < positionCount - 1)
            {
                meshTriangleIndices[index] = index;
            }
        }
    }
}
