# VR Modality Unity Project

This folder contains all the code and assets to run the VR Modality Unity
project. It has been tested with Unity version 2021.3. To run the project, open
it with Unity Hub and the appropriate Unity version, then open the scene
`Scenes/3DViewer2`.

Press 0, 1, 2, 3 to switch datasets, and spacebar to reset position/rotation of models.

To run the project on zSpace 300 hardware, the 
[zCore6 package](https://developer.zspace.com/docs/zcore-6.3/preface) is used.
To enable zSpace mode, go to the `VREngine` GameObject and change the startup
VRConfig to VRConfig_zSpace (otherwise it will use a VR simulator).