// Gmsh project created on Tue Jul 22 14:36:22 2025
//+
Point(1) = {0.0, 0.0, 0, 0.0003};
//+
Point(2) = {0.005, 0.0, 0, 0.0003};
//+
Point(3) = {0, 0.00025, 0, 0.0003};
//+
Point(4) = {0.005, 0.00025, 0, 0.0003};
//+
Line(1) = {3, 1};
//+
Line(2) = {2, 1};
//+
Line(3) = {2, 4};
//+
Line(4) = {4, 3};
//+
Curve Loop(1) = {1, -2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve("upstream_boundary", 5) = {1};
//+
Physical Curve("downstream_boundary", 6) = {3};
//+
Physical Surface("sample", 7) = {1};
