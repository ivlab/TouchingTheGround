using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.IO;

using Newtonsoft.Json.Linq;

using IVLab.OBJImport;
using IVLab.Utilities;
using IVLab.MinVR3;

[DefaultExecutionOrder(VREngine.ScriptPriority - 1)]
public class Viewer3DLoader2 : MonoBehaviour, IVREventListener, IVREventProducer
{
    public string sourcePathPrefix;
    // public string sourceFolder;
    public string[] sourceFolders;
    public string sourcePathSuffix;

    public string combinedObjFile;
    public string legendObjFile;
    public string demObjFile;
    public string mtlFile;
    public string metadataJsonFile;
    public string geoJsonFile;

    public string colormapXmlName;
    public Transform dhCursor;
    public Transform ndhCursor;
    public float modelScale = 0.001f;
    public Material objetRigurMaterial;
    public Material objetVeroBlackMaterial;

    public List<LineStudyUtils.LegendEntry> LegendEntries { get => legendEntries; }
    private List<LineStudyUtils.LegendEntry> legendEntries;
    private JObject metadata;

    // optionally constrain the objects inside some bounds
    public bool constrainObjectMovementInBounds = true;
    public float maxObjectScale = 3.0f;
    public float minObjectScale = 0.1f;
    private BoxCollider sceneBounds;

    private List<List<Vector3>> lineSegments;
    private class LinePoint
    {
        public int pointIndex;
        public Vector3 coord;
    }

    private struct DataRange<T> {
        public T min;
        public T max;
    }

    private GameObject modelParent;
    private GameObject legendParent;
    private Matrix4x4 initialModelXform;
    private Matrix4x4 initialLegendXform;
    private Vector3 previousModelScale;
    private Vector3 previousLegendScale;
    private GameObject lineRendererParent;
    // private bool hideModels = true;
    private int selectedModel = 0; // default to training

    private VREventPrototype resetEvent;
    private const string ShutdownEvent = "PhysStudyShutdown";
    private const string ResetEvent = "PhysStudyReset";

    private class GeoJsonLineFeatures
    {
        public GeoJsonLineFeature[] features;
    }
    private class GeoJsonLineFeature
    {
        public Dictionary<string, string> properties;
        public GeoJsonLineString geometry;
    }
    private class GeoJsonLineString
    {
        public string type;
        public float[][] coordinates;

        public Vector2[] GetCoordinates()
        {
            return coordinates.Select(c => new Vector2(c[0], c[1])).ToArray();
        }
    }
    private class LineConverted3D
    {
        public Vector3[] coordinates;

        public LineConverted3D(GeoJsonLineString lineString, Bounds crsExtents, Bounds unityExtents)
        {
            var coords2d = lineString.GetCoordinates();
            // assume uniform scaling
            float scaleFactor = crsExtents.extents.x / unityExtents.extents.x;
            Vector3 translation = unityExtents.center - crsExtents.center;
            List<Vector3> coords = new List<Vector3>();
            foreach (Vector2 coord in coords2d)
            {
                var coord3d = new Vector3(coord.x, 0.0f, coord.y);
                coord3d += translation;
                coord3d /= scaleFactor;
                coords.Add(coord3d);
            }

            this.coordinates = coords.ToArray();
        }
    }

    // Interrupt quit
    [RuntimeInitializeOnLoadMethod]
    static void RunOnStart()
    {
        Application.wantsToQuit += WantsToQuit;
    }

    static bool WantsToQuit()
    {
        // send a shutdown message
        VREngine.Instance.eventManager.ProcessEvent(new VREvent(ShutdownEvent));
        return true;
    }

    void OnEnable()
    {
        VREngine.Instance.eventManager.AddEventListener(this);
        resetEvent = VREventPrototype.Create(ResetEvent);
    }

    void OnDisable()
    {
        VREngine.Instance?.eventManager?.RemoveEventListener(this);
    }


    void Start()
    {
        sceneBounds = GetComponent<BoxCollider>();

        foreach (string sourceFolder in sourceFolders)
        {
            // Assumes builds are in THIS PROJECT / Builds / (some folder)
            string sourcePath = sourcePathPrefix + sourceFolder + sourcePathSuffix;
    #if !UNITY_EDITOR
            sourcePath = Path.Combine("../../", sourcePath);
    #endif

            string combinedPath = Path.Combine(sourcePath, combinedObjFile);
            string legendPath = Path.Combine(sourcePath, legendObjFile);
            string mtlPath = Path.Combine(sourcePath, mtlFile);
            string demPath = Path.Combine(sourcePath, demObjFile);
            string metadataPath = Path.Combine(sourcePath, metadataJsonFile);
            string geoJsonPath = Path.Combine(sourcePath, geoJsonFile);

            // Create parent object for this to go in
            Debug.Log("Loading object " + sourceFolder);
            GameObject parentObject = new GameObject(sourceFolder);
            parentObject.transform.SetParent(this.transform);

            // Load in the OBJ for the lines and labels and base
            OBJLoader loader = new OBJLoader();
            Debug.Log("Load combined");
            Dictionary<string, Material> combinedMaterials = new Dictionary<string, Material>
            {
                { "Light", objetRigurMaterial },
                { "Dark", objetVeroBlackMaterial },
            };
            GameObject combined = loader.LoadWithMaterials(combinedPath, combinedMaterials, false);
            Debug.Log(string.Join(", ", combined.GetComponentsInChildren<MeshRenderer>().Select(mr => string.Join("-", (mr.materials.Select(m => m.name))))));
            combined.transform.localScale = Vector3.one;
            combined.transform.SetParent(parentObject.transform);
            combined.transform.localScale *= this.modelScale;

            // Separately load in the DEM
            loader = new OBJLoader();
            Debug.Log("Load DEM");
            Dictionary<string, Material> demMaterials = new Dictionary<string, Material>
            {
                { "default", objetRigurMaterial },
            };
            GameObject dem = loader.LoadWithMaterials(demPath, demMaterials, false);
            dem.transform.localScale = Vector3.one;
            dem.transform.SetParent(parentObject.transform);
            dem.SetActive(false);
            dem.transform.localScale *= this.modelScale;

            // Load legend
            Debug.Log("Load Legend");
            loader = new OBJLoader();
            GameObject legend = loader.LoadWithMaterials(legendPath, combinedMaterials, false);
            legend.transform.localScale = Vector3.one;
            legend.transform.SetParent(parentObject.transform);
            legend.transform.localScale *= this.modelScale;
            legend.transform.localRotation = Quaternion.LookRotation(Vector3.right, Vector3.up);

            // Load in JSON
            metadata = null;
            using (StreamReader reader = new StreamReader(metadataPath))
            {
                string metadataText = reader.ReadToEnd();
                metadata = JObject.Parse(metadataText);
            }

            // Get model scale from JSON
            Vector3 modelScale = metadata["scale"].ToObject<Vector3>();
            // convert to Z up
            modelScale = new Vector3(modelScale.x, modelScale.z, modelScale.y);

            legendEntries = metadata["legendEntries"].ToObject<List<LineStudyUtils.LegendEntry>>();
            float legendHeight = metadata["config"]["legend_platform_height"].ToObject<System.Single>();

            // Get elevation range from JSON
            List<float> demRange = metadata["demElevationRange"].ToObject<List<float>>();
            DataRange<float> elevationRange = new DataRange<float>();
            elevationRange.min = demRange[0];
            elevationRange.max = demRange[1];


            // Load in geojson for line rendering
            GeoJsonLineFeatures geojson = null;
            using (StreamReader reader = new StreamReader(geoJsonPath))
            {
                string geoJsonText = reader.ReadToEnd();
                JObject jo = JObject.Parse(geoJsonText);
                geojson = jo.ToObject<GeoJsonLineFeatures>();
            }
            // proceed later in the script where we've already defined DEM bounds in Unity...

            // load data
            string dataPath = "USGS/DEM/KeyData/" + dem.name;

            Mesh heightMesh = dem.GetComponentInChildren<MeshFilter>().sharedMesh;

            List<float> heights = new List<float>();
            float minHeight = heightMesh.vertices.Select(v => v.y).Min();
            float maxHeight = heightMesh.vertices.Select(v => v.y).Max();
            foreach (Vector3 vert in heightMesh.vertices)
            {
                heights.Add(vert.y / modelScale.y);
            }


            // Create GameObject for all model components
            modelParent = new GameObject("Model");
            modelParent.transform.SetParent(parentObject.transform);
            // group.GroupRoot.transform.SetParent(modelParent.transform);
            combined.transform.SetParent(modelParent.transform);
            modelParent.transform.localPosition = new Vector3(0.0f, -0.05f, 0.0f);
            initialModelXform = modelParent.transform.localToWorldMatrix;

            BoxCollider modelCollider = modelParent.AddComponent<BoxCollider>();
            MeshFilter modelMesh = modelParent.GetComponentsInChildren<MeshFilter>().First(mf => mf.gameObject.name.Contains(sourceFolder));
            modelCollider.size = modelMesh.transform.localToWorldMatrix.MultiplyVector(modelMesh.mesh.bounds.size);
            modelCollider.center = modelMesh.mesh.bounds.center * this.modelScale;

            // Create a scaled version of the DEM data mesh
            GameObject scaledDemMesh = new GameObject("Scaled DEM Mesh");
            scaledDemMesh.transform.SetParent(modelParent.transform, false);
            scaledDemMesh.transform.localScale = Vector3.one * this.modelScale;
            MeshCollider demCollider = scaledDemMesh.AddComponent<MeshCollider>();
            demCollider.sharedMesh = dem.GetComponentInChildren<MeshFilter>().sharedMesh;

            // get data extents in DEM space (EPSG:3857) and convert all line segments into Unity format
            float[][] crsExtents = metadata["extents"].ToObject<float[][]>();
            Vector3 minExtent = new Vector3(crsExtents[0][0], 0.0f, crsExtents[0][1]);
            Vector3 maxExtent = new Vector3(crsExtents[1][0], 0.0f, crsExtents[1][1]);
            Vector3 center = (maxExtent + minExtent) / 2.0f;
            Vector3 extent = (maxExtent - minExtent);
            Bounds crsBounds = new Bounds(center, extent);

            // Create line renderers to render overlay geometry
            lineRendererParent = new GameObject("Line Renderers");
            lineRendererParent.transform.SetParent(modelParent.transform, false);

            // Material lrMat = new Material(Shader.Find("IVLab/UnlitLineRenderer"));
            // lrMat.color = Color.black;
            const float LineOverlayCastHeight = 0.1f; // meters
            const float AboveSurfaceY = 0.0001f; // meters to prevent z fighting

            for (int l = 0; l < geojson.features.Length; l++)
            {
                var line = geojson.features[l];
                LineConverted3D line0 = new LineConverted3D(line.geometry, crsBounds, demCollider.bounds);

                // raycast down onto a mesh collider for the DEM to get the actual
                // (projected to 3d) coordinates. Lerp between each vertex to get a
                // suitable number of sub-vertices.
                // want 0.1mm segments.
                const float MinSegmentLength = 0.0001f;
                List<Vector3> subVertices = new List<Vector3>();
                for (int v = 0; v < line.geometry.coordinates.Length - 1; v++)
                {
                    Vector3 vert = line0.coordinates[v];
                    Vector3 vertNext = line0.coordinates[v + 1];
                    float dist = (vertNext - vert).magnitude;
                    int numSegments = Mathf.RoundToInt(dist / MinSegmentLength);
                    float step = 1.0f / numSegments;
                    for (float t = 0; t < 1.0f; t += step)
                    {
                        Vector3 intermediate = Vector3.Lerp(vert, vertNext, t);

                        // find z coord by raycast
                        Ray r = new Ray(intermediate + Vector3.up * LineOverlayCastHeight, Vector3.down);
                        RaycastHit hit;
                        if (demCollider.Raycast(r, out hit, 1.0f))
                        {
                            intermediate.y = hit.point.y - demCollider.transform.position.y + AboveSurfaceY;
                            subVertices.Add(intermediate);
                        }
                    }
                }

                if (subVertices.Count() >= 2)
                {
                    GameObject lro = new GameObject("Line Render Object " + l);
                    lro.transform.SetParent(lineRendererParent.transform, false);
                    FixedNormalLineRenderer lr = lro.AddComponent<FixedNormalLineRenderer>();
                    lr.lineNormal = Vector3.up;
                    lr.material.color = Color.black;
                    lr.width = metadata["config"]["line_width"].ToObject<Single>() / 1000.0f;
                    lr.positionCount = subVertices.Count;
                    lr.SetPositions(subVertices.ToArray());
                    lr.RecalculateGeometry();
                }
            }

            // don't need the DEM collider anymore
            Destroy(scaledDemMesh);


            // Create GameObject for legend. Initially place it to the right of the model
            legendParent = new GameObject("Legend");
            legendParent.transform.SetParent(parentObject.transform);
            legend.transform.SetParent(legendParent.transform);
            float demSizeX = dem.GetComponentInChildren<MeshFilter>().mesh.bounds.extents.x * dem.transform.localScale.x;
            float legendSizeX = legend.GetComponentInChildren<MeshFilter>().mesh.bounds.extents.x  * legend.transform.localScale.x;
            float legendSizeY = legend.GetComponentInChildren<MeshFilter>().mesh.bounds.extents.y  * legend.transform.localScale.y;
            legendParent.transform.localPosition = new Vector3(demSizeX + legendSizeX / 4.0f, -legendSizeY - 0.051f, -0.05f);
            initialLegendXform = legendParent.transform.localToWorldMatrix;


            BoxCollider legendCollider = legendParent.AddComponent<BoxCollider>();
            MeshFilter legendMesh = legendParent.GetComponentInChildren<MeshFilter>();
            Bounds legendBaseAndbars = new Bounds(legendMesh.mesh.bounds.center, legendMesh.mesh.bounds.size);
            // legendBaseAndbars.Encapsulate(cubeMesh.bounds);
            var size = legendMesh.transform.localToWorldMatrix.MultiplyVector(legendBaseAndbars.size);
            size.x = size.x > 0 ? size.x : -size.x;
            size.y = size.y > 0 ? size.y : -size.y;
            size.z = size.z > 0 ? size.z : -size.z;
            legendCollider.size = size;
            legendCollider.center = legendBaseAndbars.center * this.modelScale;

        }

        // select first model
        for (int g = 0; g < this.transform.childCount; g++)
        {
            this.transform.GetChild(g).gameObject.SetActive(false);
        }
        this.transform.GetChild(selectedModel).gameObject.SetActive(true);

        // change to new model parent
        modelParent = this.transform.GetChild(selectedModel).Find("Model").gameObject;
        legendParent = this.transform.GetChild(selectedModel).Find("Legend").gameObject;
    }

    void Update()
    {
        // Sync the scale between the legend and model (model takes precedence)
        if (!Mathf.Approximately(legendParent.transform.localScale.magnitude, previousLegendScale.magnitude))
        {
            modelParent.transform.localScale = legendParent.transform.localScale;
        }
        if (!Mathf.Approximately(modelParent.transform.localScale.magnitude, previousModelScale.magnitude))
        {
            legendParent.transform.localScale = modelParent.transform.localScale;
            foreach (FixedNormalLineRenderer lr in lineRendererParent.GetComponentsInChildren<FixedNormalLineRenderer>())
            {
                lr.width = metadata["config"]["line_width"].ToObject<Single>() / (1000.0f * modelParent.transform.localScale.x);
                lr.RecalculateGeometry();
            }
        }

        // Make sure objects are staying in bounds, if desired
        if (constrainObjectMovementInBounds)
        {
            if (!sceneBounds.bounds.Contains(modelParent.transform.position))
            {
                modelParent.transform.position = sceneBounds.bounds.ClosestPoint(modelParent.transform.position);
            }
            if (!sceneBounds.bounds.Contains(legendParent.transform.position))
            {
                legendParent.transform.position = sceneBounds.bounds.ClosestPoint(legendParent.transform.position);
            }
            float modelParentScale = modelParent.transform.localScale.x;
            float legendParentScale = legendParent.transform.localScale.x;
            modelParentScale = Mathf.Clamp(modelParentScale, minObjectScale, maxObjectScale);
            legendParentScale = Mathf.Clamp(legendParentScale, minObjectScale, maxObjectScale);
            modelParent.transform.localScale = Vector3.one * modelParentScale;
            legendParent.transform.localScale = Vector3.one * legendParentScale;
        }

        previousModelScale = modelParent.transform.localScale;
        previousLegendScale = legendParent.transform.localScale;

        // Handle appear/disappear and reset between trials
        if (Input.GetKeyDown(KeyCode.Space))
        {
            ResetModels();
        }

        // Handle changing which GameObject is active
        int lastSelected = selectedModel;
        if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            selectedModel = 0;
        }
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            selectedModel = 1;
        }
        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            selectedModel = 2;
        }
        if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            selectedModel = 3;
        }
        if (lastSelected != selectedModel)
        {
            for (int g = 0; g < this.transform.childCount; g++)
            {
                this.transform.GetChild(g).gameObject.SetActive(false);
            }

            // reset the old model parent
            modelParent.transform.FromMatrix(initialModelXform);
            legendParent.transform.FromMatrix(initialLegendXform);

            // change to new model parent
            modelParent = this.transform.GetChild(selectedModel).Find("Model").gameObject;
            legendParent = this.transform.GetChild(selectedModel).Find("Legend").gameObject;

            // reset the new model parent
            modelParent.transform.FromMatrix(initialModelXform);
            legendParent.transform.FromMatrix(initialLegendXform);

            this.transform.GetChild(selectedModel).gameObject.SetActive(true);
        }
    }

    private void ResetModels()
    {
        modelParent.transform.FromMatrix(initialModelXform);
        legendParent.transform.FromMatrix(initialLegendXform);
    }

    private float MapRange(float value, float low1, float high1, float low2, float high2)
    {
        return low2 + (high2 - low2) * (value - low1) / (high1 - low1);
    }

    public List<IVREventPrototype> GetEventPrototypes()
    {
        return new List<IVREventPrototype>
        {
            VREventPrototype.Create(ShutdownEvent)
        };
    }

    public void OnVREvent(VREvent evt)
    {
        if (evt.Matches(resetEvent))
        {
            ResetModels();
        }
    }

    public void StartListening() {}
    public void StopListening() {}
}
