# Physicalization Study 1

Companion code to VIS 2023 paper comparing digital and physical spatial data visualizations

## Prerequisites

- Blender 3.3 (may work with other versions, untested)
- [Blender GIS plugin](https://github.com/domlysz/BlenderGIS/releases) installed
- [Blender 3MF plugin](https://github.com/Ghostkeeper/Blender3mfFormat/releases) installed
    - can be found under "testing" plugins
- QGIS installed
- DEM data downloaded from USGS

## The process

The process for creating these datasets seems to be getting ever more
complicated. As such, here's a quick overview (and details are below):

- [Physicalization Study 1](#physicalization-study-1)
  - [Prerequisites](#prerequisites)
  - [The process](#the-process)
  - [1. Import the DEM to QGIS](#1-import-the-dem-to-qgis)
  - [2. Import Web Mercator DEM to Blender](#2-import-web-mercator-dem-to-blender)
  - [3. Find an appropriate scale](#3-find-an-appropriate-scale)
  - [4. Segment the DEM into squares](#4-segment-the-dem-into-squares)
  - [5. Shorten the datasets](#5-shorten-the-datasets)
  - [6. Generate task answers and geometry](#6-generate-task-answers-and-geometry)
    - [Run the script](#run-the-script)
    - [Post Processing](#post-processing)
      - [Materials](#materials)
      - [Solidify DEM](#solidify-dem)
      - [Object Centering](#object-centering)
      - [Crosshairs](#crosshairs)
      - [Labels](#labels)
      - [Legends](#legends)
  - [7. Prepare for Unity 3D Visualization](#7-prepare-for-unity-3d-visualization)
  - [8. Prepare for QGIS 2D Visualization](#8-prepare-for-qgis-2d-visualization)
    - [Crop DEM to the same dimensions as Blender output and shorten heights](#crop-dem-to-the-same-dimensions-as-blender-output-and-shorten-heights)
    - [Create contours](#create-contours)
    - [Import lines](#import-lines)
  - [9. Prepare for 3D Printing](#9-prepare-for-3d-printing)
    - [Common steps](#common-steps)
    - [Special Steps for FDM in-lab printing](#special-steps-for-fdm-in-lab-printing)
    - [Special Steps for PolyJet printing service](#special-steps-for-polyjet-printing-service)



## 1. Import the DEM to QGIS

1. open QGIS and load an OpenStreetMap basemap layer
2. [download geotiff from USGS](https://apps.nationalmap.gov/downloader/) (select area and use 3DEP, 1-degree, current)
3. import to QGIS
4. save layer as (re-export geotiff, convert to web mercator (EPSG:3857)


## 2. Import Web Mercator DEM to Blender

1. import to Blender GIS (georeferenced raster, DEM raw data build, step=5)
2. give an initial scale (0.001 in x/y and 0.01 in z [10x vertical exagg])
3. adjust viewport clip start/end (press N > View), something like 10, 10000 is usually good


## 3. Find an appropriate scale

This step is a little weird. It involves repeatedly poking a paraboloid (virtual
finger) down on the dataset to see if there are any valleys that a human finger
wouldn't be able to touch. The goal is to make sure the dataset is at a scale
such that every point is touchable.

1. In the Scripting panel, open the `dataset_isvalid.py` script in this folder.
2. Open the Blender console (Window > Toggle System Console)
3. Make sure the DEM object is selected, then run the script (Ctrl-P)
4. Observe the script's output (in the console) of the % of vertices are invalid.
5. Rescale the data so there falls within a desirable range (ideally, less than 10% invalid)
6. Repeat Steps 3-5 as necessary
7. Can be helpful to insert a fingertip-size cube in the scene

I found that a scale of (0.005, 0.005, 0.025) is actually pretty good (at least
for the Mt Whitney dataset). This is a 5x vertical exaggeration.


## 4. Segment the DEM into squares

1. In the Scripting panel, open the `find_ranges.py` script in this folder.
2. Open the Blender console (Window > Toggle System Console)
3. Make sure the DEM object is selected, then run the script (Ctrl-P)
4. Duplicate the DEM
4. Go into edit mode for the copied DEM
5. Select vertices inside the subset you wish to use
6. Select inverse (Ctrl I) and delete all the other verts
7. This should leave you with a nice, square subset (check dimensions to make sure it's ~150x150mm)
8. Repeat this process for all the subsets you wish to keep, organizing them all into a Blender Collection

## 5. Shorten the datasets

The datasets sometimes are ridiculously tall for 3d printing. As such, we
shorten them so the original data minimum is at a fixed height for all datasets
(for example, 1mm above the origin).

1. In the scripting panel, open the `shorten_dataset.py` script in this folder
2. Open the Blender console window
3. Select the Blender collection of subsets you created in [Part 4](#4-segment-the-dem-into-squares).
4. Run the script (Ctrl-P)
5. Check out the output:
    - New Blender collection 'Shortened' - copy/paste these into their own
    collections to create the final datasets
    - 'shortened.json' file that was created - contains the CRS extents and the
    amount each dataset was shortened by. Can be use in [QGIS to get the correct
    subset and height](#8-prepare-for-qgis-2d-visualization).


## 6. Generate task answers and geometry

### Run the script

Open the `generate_crosshair_geometry.py` script in Blender.

Make sure the DEM object is selected (should be in its own collection, copied from the previous Part)

It may be helpful to toggle system console (Window > Toggle System Console) to see script output.

Hit alt+p to run the script.

Tweak script configuration as necessary, they will be saved in metadata.json.


### Post Processing

#### Materials
1. Apply correct Light/Dark materials to all objects

#### Solidify DEM
1. Make a copy of the DEM (call it "something"_solid)
2. Select all boundary vertices
    - May need to use "select similar"
3. Extrude down z to global z=0
4. Fill face
5. Make another copy of DEM, call it "something"_dem (for Unity)

#### Object Centering

1. On the solid DEM, Ctrl+Shift+Alt+C and set "Origin to Geometry" (use center of bounding box, it's the default)
2. Shift+S and do Cursor to Selection
3. Then, set all objects' (labels, crosshairs, dem) origins to that object using Ctrl+Shift+Alt+C / Set to 3D cursor
4. Set the 3D cursor back to the world origin
5. Selection to Cursor

#### Crosshairs

1. Make a copy of crosshairs object
2. Do an INTERSECT modifier with the original heightmap (may need to copy heightmap and INTERSECT crosshairs instead)
3. Delete/clean up any extraneous geometry not on the surface of the heightmap
4. Assign a vertex group to all vertices (maybe call it "lower" -- will make it easier to reclaim this set of vertices later)
5. Move all vertices DOWN in z 0.3mm
6. Extrude UP in z 0.6mm (leaves 0.3mm above surface)
    - not too obvious, any lower and it disappears in slicing
    - ONLY do this for FDM single-color 3d printing

#### Labels
1. Apply the UNION modifiers on the label_base object
2. Do a SUBTRACT modifier with label_base - label_platforms (cutaway)
3. Do (or don't, probably) a UNION with label_base + label_text

#### Legends

1. At a minimum, apply UNION to legend_base and legend_bars


## 7. Prepare for Unity 3D Visualization

1. Ensure proper materials are assigned
2. In the DEM, shade the top faces smooth (Edit Mode > select top faces > Face > Shade Smooth)
3. Export OBJs - all have +Z Forward
    - full.obj (contains solid DEM + joined labels)
      - sometimes need to merge verts by distance 0.01m to get it to show up right in Unity
    - dem.obj (contains only the "something"_dem mesh)
    - legend.obj (contains)

## 8. Prepare for QGIS 2D Visualization

### Crop DEM to the same dimensions as Blender output and shorten heights

1. Consult shortened.json `extent`. Do a Raster > Clip Raster By Extent and paste in the extents. Save file.
2. Do a Raster Calculator on the clipped layer to shorten:
    - operation would be `some_layer - [amount pasted in from 'height_difference' in shortened.json]`
    - save as file in the `vis` folder - this will be the final DEM the user sees.
3. You can remove the original DEM and tmp layers from the project

### Create contours

1. Go to *Raster > Extraction > Contours*.
2. Set the contour interval to match the legend generated by Blender
3. Apply styling contour_style.qml
4. Save/export as contours.gpkg
5. Repeat with intermediate/between contours (interval above / 5, and use
contour_style_between.qml)

### Import lines

1. Drag and drop the `metadata.geojson` file in the root of this dataset to create a new layer
2. Apply styling line_style_labels.qml


## 9. Prepare for 3D Printing

### Common steps

1. Make a duplicate of the Solid model

### Special Steps for FDM in-lab printing

1. Join everything (labels, crosshairs, solidified DEM) together using UNION modifiers.
    - under Solver Options, set Materials to "Transfer"
2. Export the joined geometry as an STL file

### Special Steps for PolyJet printing service

1. Duplicate the solid model and the crosshairs model
2. Apply any outstanding UNION modifiers (e.g., with label base)
3. On the crosshairs, Move the 0.30mm label text and line extrusions down so they're ALMOST
touching the surface (probably 0.01mm above surface is fine)
4. Move the lower vertices of the line extrusions down at least 1mm (will help with difference)
5. Do a DIFFERENCE modifier with the
3. Apply scale (Ctrl+A)
5. Export joined geometry as a .3mf file (requires [Blender 3mf exporter](https://github.com/Ghostkeeper/Blender3mfFormat/releases))
