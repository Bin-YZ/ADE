//+
Point(1) = {0.0, 0.0, 0, 0.0001};
//+
Point(2) = {0.012, 0.0, 0, 0.0001};
//+
Point(3) = {0.012, 0.0001, 0, 0.0001};
//+
Point(4) = {0.011, 0.0001, 0, 0.0001};
//+
Point(5) = {0.011, 0.0, 0, 0.0001};
//+
Point(6) = {0.001, 0.0, 0, 0.0001};
//+
Point(7) = {0.001, 0.0001, 0, 0.0001};
//+
Point(8) = {0.0, 0.0001, 0, 0.0001};
//+
Line(1) = {1, 6};
//+
Line(2) = {6, 5};
//+
Line(3) = {5, 2};
//+
Line(4) = {2, 3};
//+
Line(5) = {3, 4};
//+
Line(6) = {4, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 1};
//+
Line(9) = {6, 7};
//+
Line(10) = {5, 4};
//+
Curve Loop(1) = {1, 9, 7, 8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, 10, 6, -9};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {3, 4, 5, -10};
//+
Plane Surface(3) = {3};
//+
Physical Surface("filter", 11) = {1, 3};
//+
Physical Surface("sample", 12) = {2};
//+
Physical Curve("left_boundary", 13) = {8};
//+
Physical Curve("right_boundary", 14) = {4};
//+
Physical Curve("closed_boundary", 15) = {5, 3, 2, 6, 1, 7};
