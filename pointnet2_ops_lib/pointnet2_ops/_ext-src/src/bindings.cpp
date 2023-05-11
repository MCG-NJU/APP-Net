#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points", &gather_points);
  m.def("gather_points_grad", &gather_points_grad);
  m.def("furthest_point_sampling", &furthest_point_sampling);

  m.def("three_nn", &three_nn);
  m.def("three_interpolate", &three_interpolate);
  m.def("three_interpolate_grad", &three_interpolate_grad);

  m.def("ball_query", &ball_query);
  


  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);
  m.def("one_nn", &one_nn);
  m.def("one_nn_pf", &one_nn_pf);
  m.def("one_interpolate", &one_interpolate);
  m.def("one_interpolate_grad", &one_interpolate_grad);
  m.def("chamfer_distance", &chamfer_distance);
  m.def("chamfer_distance_radius", &chamfer_distance_radius);
}
