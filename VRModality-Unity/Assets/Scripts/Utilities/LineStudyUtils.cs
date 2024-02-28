using UnityEngine;
using System.Collections.Generic;

public static class LineStudyUtils
{
    public class LegendEntry
    {
        public float value;
        public Vector3 scale;
        public Vector3 position;

        public override string ToString()
        {
            return $"LegendEntry({value}, {scale}, {position})";
        }
    }
}