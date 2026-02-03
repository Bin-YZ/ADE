// Gmsh project created on Tue Jul 22 14:36:22 2025
//+
Point(1) = {0.0, 0.0, 0, 0.0003};
//+
Point(2) = {0.005, 0.0, 0, 0.0003};
//+
Line(1) = {2, 1};
//+
Physical Point("upstream", 3) = {1};
//+
Physical Point("downstream", 4) = {2};
//+
Physical Curve("sample", 5) = {1};
