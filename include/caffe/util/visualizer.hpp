#ifndef CAFFE_UTIL_VISUALIZER_H_
#define CAFFE_UTIL_VISUALIZER_H_

#include "caffe/common.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
class Visualizer {
 public:
  virtual void CreateLogRecord(shared_ptr<Net<Dtype> > net,
      LogRecord* log_record) = 0;
};

}  // namespace caffe

#endif  // CAFFE_UTIL_VISUALIZER_H_
