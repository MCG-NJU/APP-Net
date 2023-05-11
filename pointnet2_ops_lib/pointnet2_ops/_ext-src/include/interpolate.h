#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows);
at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight);
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m);

std::vector<at::Tensor> one_nn(at::Tensor unknowns, at::Tensor knows);
at::Tensor one_nn_pf(at::Tensor unknowns, at::Tensor knows);
at::Tensor one_interpolate(at::Tensor points, at::Tensor idx);
//void rand_chamfer_distance(torch::Tensor xyz1, torch::Tensor xyz2,
//                           torch::Tensor dists1, torch::Tensor dists1_i, torch::Tensor start_idx, const float th);
at::Tensor one_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  const int m);
void chamfer_distance(torch::Tensor xyz1, torch::Tensor xyz2,
								   torch::Tensor dists1, torch::Tensor dists1_i);
void chamfer_distance_radius(torch::Tensor xyz1, torch::Tensor xyz2, const float radius,
								   torch::Tensor dists1, torch::Tensor dists1_i);

