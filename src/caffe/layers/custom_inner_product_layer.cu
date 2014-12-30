#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iostream>
#include <iomanip>

using std::cin;
using std::cout;
using std::endl;

namespace caffe {

template <typename Dtype>
void CustomInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  custom_bottom_.ReshapeLike (*bottom[0]);
  custom_bottom_.CopyFrom(*bottom[0]);  // Copy bottom data
  // Set specific bottom data as specific value
  unsigned int fixed_point_size = fixed_point_.size();
  Dtype* p_custom_bottom_ = custom_bottom_.mutable_cpu_data();

//  LOG(INFO) << "K: " << K_;
//  LOG(INFO) << "num: " << bottom[0]->num();
//  LOG(INFO) << "channels: " << bottom[0]->channels();
  const int num = bottom[0]->num();

  for(int i = 0; i < num; ++i) {
    for(unsigned int j = 0; j < fixed_point_size; ++j) {
      p_custom_bottom_[ (i * K_ + fixed_point_.at(j)) ] = fixed_value_;
    }
  }

//   // // Test if the results are correct
//   std::cout << std::setprecision(2);
//   std::cout << std::fixed;
//   for (int j = 0; j < custom_bottom_.channels(); ++j ) {
//     cout << custom_bottom_.data_at(0, j, 0, 0) << " ";
//   }
//   cout << endl;
  
  const Dtype* bottom_data = custom_bottom_.gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void CustomInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)0.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_CLASS(CustomInnerProductLayer);

}  // namespace caffe
