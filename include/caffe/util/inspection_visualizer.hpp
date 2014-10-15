#ifndef CAFFE_UTIL_INSPECTION_VISUALIZER_H_
#define CAFFE_UTIL_INSPECTION_VISUALIZER_H_

#include "caffe/util/visualizer.hpp"

namespace caffe {

template <typename Dtype>
class InspectionVisualizer : public Visualizer<Dtype> {
 public:
  void CreateLogRecord(shared_ptr<Net<Dtype> > net,
      LogRecord* log_record);
};

}  // namespace caffe

#endif  // CAFFE_UTIL_INSPECTION_VISUALIZER_H_
