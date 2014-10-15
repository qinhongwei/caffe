#ifndef CAFFE_UTIL_BASIC_VISUALIZER_H_
#define CAFFE_UTIL_BASIC_VISUALIZER_H_

#include <string>

#include "caffe/util/visualizer.hpp"

namespace caffe {

template <typename Dtype>
class BasicVisualizer : public Visualizer<Dtype> {
 public:
  void CreateLogRecord(shared_ptr<Net<Dtype> > net,
      LogRecord* log_record);

 protected:
  void ExtractBlob(const string& name, const Blob<Dtype>* blob, int index,
      BlobSnapshot* message);

  static const int kMaxExamples;
  static const int kMaxWidth = 263;
};

}  // namespace caffe

#endif  // CAFFE_UTIL_BASIC_VISUALIZER_H_
