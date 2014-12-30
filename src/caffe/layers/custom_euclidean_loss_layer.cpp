#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iostream>
#include <iomanip>

using std::cin;
using std::cout;
using std::endl;


namespace caffe {

template <typename Dtype>
void CustomEuLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  /**
   * bottom[0]: predicted
   * bottom[1]: ground truth
   */
  const CustomEuLossParameter& param = this->layer_param_.custom_eu_loss_param();
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  custom_label_.Reshape(bottom[0]->num(), bottom[0]->channels(),
  	      bottom[0]->height(), bottom[0]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  is_selected_ = param.is_selected();

  if (is_selected_) {
    selected_point_.clear();
    std::copy(param.selected_point().begin(),
    		  param.selected_point().end(),
        		std::back_inserter(selected_point_));
    CHECK_EQ(bottom[0]->channels(), selected_point_.size());

    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const unsigned int selected_point_size = selected_point_.size();
    Dtype* p_custom_label_ = custom_label_.mutable_cpu_data();

	  for(int i = 0; i < num; ++i) {
		for(unsigned int j = 0; j < selected_point_size; ++j) {
		  p_custom_label_[ (i * channels + j) ] = bottom[1]->data_at(i, selected_point_.at(j), 0, 0);
		}
	  }

//	   // // Test if the results are correct
//	   std::cout << std::setprecision(2);
//	   std::cout << std::fixed;
//	   LOG(INFO) << "Selected Label";
//	   for (int j = 0; j < bottom[1]->channels(); ++j ) {
//	     cout << bottom[1]->data_at(0, j, 0, 0) << " ";
//	   }
//	   cout << endl;
//
//	   // // Test if the results are correct
//	   std::cout << std::setprecision(2);
//	   std::cout << std::fixed;
//	   for (int j = 0; j < custom_label_.channels(); ++j ) {
//	     cout << custom_label_.data_at(0, j, 0, 0) << " ";
//	   }
//	   cout << endl;

  } else {
	CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  }
}

template <typename Dtype>
void CustomEuLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      custom_label_.cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CustomEuLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
//		LOG(INFO) << "Propagate down " << i;
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();
      caffe_cpu_axpby(
          (*bottom)[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CustomEuLossLayer);
#endif

INSTANTIATE_CLASS(CustomEuLossLayer);

}  // namespace caffe
