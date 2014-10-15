#include <algorithm>
#include <string>
#include <vector>

#include "caffe/util/inspection_visualizer.hpp"

namespace caffe {

template <typename Dtype>
void InspectionVisualizer<Dtype>::CreateLogRecord(shared_ptr<Net<Dtype> > net,
    LogRecord* log_record) {
  log_record->set_type(LogRecord::INSPECTION);

  InspectionRecord* inspection_record = log_record->MutableExtension(
      InspectionRecord::parent);

  const vector<shared_ptr<Layer<Dtype> > >& layers = net->layers();
  typedef typename vector<shared_ptr<Layer<Dtype> > >::const_iterator LayerIter;
  for (LayerIter layer = layers.begin(); layer != layers.end(); ++layer) {
    const vector<shared_ptr<Blob<Dtype> > >& blobs = (*layer)->blobs();

    const string& name = (*layer)->layer_param().name();

    for (size_t blob_index = 0; blob_index < blobs.size(); ++blob_index) {
      shared_ptr<Blob<Dtype> > blob = blobs.at(blob_index);

      InspectionResult* inspection_result =
          inspection_record->add_inspection_results();

      inspection_result->set_layer_name(name);
      inspection_result->set_layer_index(blob_index);

      vector<Dtype> values(blob->cpu_data(), blob->cpu_data() + blob->count());
      sort(values.begin(), values.end());

      size_t pct15_index = values.size() * 0.15;
      size_t pct50_index = values.size() * 0.50;
      size_t pct85_index = values.size() * 0.85;

      CHECK_LT(0, values.size());

      Dtype value0 = values.front();
      Dtype value15 = values.at(pct15_index);
      Dtype value50 = values.at(pct50_index);
      Dtype value85 = values.at(pct85_index);
      Dtype value100 = values.back();

      vector<Dtype> grad(blob->cpu_diff(), blob->cpu_diff() + blob->count());
      sort(grad.begin(), grad.end());

      CHECK_LT(0, grad.size());

      Dtype grad0 = grad.front();
      Dtype grad15 = grad.at(pct15_index);
      Dtype grad50 = grad.at(pct50_index);
      Dtype grad85 = grad.at(pct85_index);
      Dtype grad100 = grad.back();

      inspection_result->set_weight_0pct(value0);
      inspection_result->set_weight_15pct(value15);
      inspection_result->set_weight_50pct(value50);
      inspection_result->set_weight_85pct(value85);
      inspection_result->set_weight_100pct(value100);

      inspection_result->set_grad_0pct(grad0);
      inspection_result->set_grad_15pct(grad15);
      inspection_result->set_grad_50pct(grad50);
      inspection_result->set_grad_85pct(grad85);
      inspection_result->set_grad_100pct(grad100);
    }
  }
}

INSTANTIATE_CLASS(InspectionVisualizer);

}  // namespace caffe
