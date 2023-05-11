#include "interpolate.h"
#include "utils.h"

void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx);
void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out);
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points);
void one_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx);
void one_nn_pf_kernel_wrapper(int b, int n, int m, int c, const float *unknown,
                             const float *known, int *idx);
void one_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx, float *out);
void one_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, float *grad_points);
void chamfer_distance_kernel_wrapper(const float* xyz1, const float* xyz2,
											 int n_batch, int n_points_1, int n_points_2,
											 float* dists1, int* dists1_i);

void chamfer_distance_radius_kernel_wrapper(const float* xyz1, const float* xyz2, const float radius,
											 int n_batch, int n_points_1, int n_points_2,
											 float* dists1, int* dists1_i);

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  if (unknowns.is_cuda()) {
    CHECK_CUDA(knows);
  }

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  if (unknowns.is_cuda()) {
    three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                            unknowns.data_ptr<float>(), knows.data_ptr<float>(),
                            dist2.data_ptr<float>(), idx.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return {dist2, idx};
}

at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (points.is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    three_interpolate_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), idx.size(1),
        points.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(),
        output.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (grad_out.is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), m},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.is_cuda()) {
    three_interpolate_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out.data_ptr<float>(), idx.data_ptr<int>(),
        weight.data_ptr<float>(), output.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

std::vector<at::Tensor> one_nn(at::Tensor unknowns, at::Tensor knows) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  if (unknowns.is_cuda()) {
    CHECK_CUDA(knows);
  }

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 1},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(1), 1},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  if (unknowns.is_cuda()) {
    one_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                            unknowns.data_ptr<float>(), knows.data_ptr<float>(),
                            dist2.data_ptr<float>(), idx.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return {dist2, idx};
}

at::Tensor one_nn_pf(at::Tensor unknowns, at::Tensor knows) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  if (unknowns.is_cuda()) {
    CHECK_CUDA(knows);
  }

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 1},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));

  if (unknowns.is_cuda()) {
    one_nn_pf_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1), knows.size(2),
                            unknowns.data_ptr<float>(), knows.data_ptr<float>(), idx.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx;
}

at::Tensor one_interpolate(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    one_interpolate_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), idx.size(1),
        points.data_ptr<float>(), idx.data_ptr<int>(),
        output.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
at::Tensor one_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  const int m) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), m},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.is_cuda()) {
    one_interpolate_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out.data_ptr<float>(), idx.data_ptr<int>(),
        output.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

void chamfer_distance(torch::Tensor xyz1, torch::Tensor xyz2,
								   torch::Tensor dists1, torch::Tensor dists1_i) {
  CHECK_CONTIGUOUS(xyz1);
  CHECK_CUDA(xyz1);
	CHECK_IS_FLOAT(xyz1);

	CHECK_CONTIGUOUS(xyz2);
  CHECK_CUDA(xyz2);
	CHECK_IS_FLOAT(xyz2);

	chamfer_distance_kernel_wrapper(xyz1.data_ptr<float>(), xyz2.data_ptr<float>(),
											xyz1.size(0), xyz1.size(1), xyz2.size(1),
											dists1.data_ptr<float>(), dists1_i.data_ptr<int>());
}

void chamfer_distance_radius(torch::Tensor xyz1, torch::Tensor xyz2, const float radius,
								   torch::Tensor dists1, torch::Tensor dists1_i) {
  CHECK_CONTIGUOUS(xyz1);
  CHECK_CUDA(xyz1);
	CHECK_IS_FLOAT(xyz1);
  // CHECK_IS_FLOAT(radius);

	CHECK_CONTIGUOUS(xyz2);
  CHECK_CUDA(xyz2);
	CHECK_IS_FLOAT(xyz2);

	chamfer_distance_radius_kernel_wrapper(xyz1.data_ptr<float>(), xyz2.data_ptr<float>(), radius,
											xyz1.size(0), xyz1.size(1), xyz2.size(1),
											dists1.data_ptr<float>(), dists1_i.data_ptr<int>());
}
// end{lcx}