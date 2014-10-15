#include <opencv2/opencv.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/util/basic_visualizer.hpp"

namespace caffe {

template <typename Dtype>
const int BasicVisualizer<Dtype>::kMaxExamples = 3;

template <typename Dtype>
void BasicVisualizer<Dtype>::ExtractBlob(const string& name,
    const Blob<Dtype>* blob, int index, BlobSnapshot* message) {
  message->set_layer_name(name);
  message->set_layer_index(index);

  Dtype min_value = *std::min_element(blob->cpu_data(),
      blob->cpu_data() + blob->count());
  Dtype max_value = *std::max_element(blob->cpu_data(),
      blob->cpu_data() + blob->count());

  int num_examples = std::min(kMaxExamples, blob->num());
  int num_channels = (blob->channels() == 3) ? 3 : 1;

  cv::Mat image(blob->height() + 2, (blob->width() + 1) * num_examples + 1,
      num_channels == 3 ? CV_8UC3 : CV_8UC1, cv::Scalar(0));
  const Dtype* src1 = blob->cpu_data();
  unsigned char* dst1 = image.ptr();
  for (int num = 0; num < num_examples; ++num) {
    const Dtype* src2 = src1 +
        num * blob->channels() * blob->height() * blob->width();
    unsigned char* dst2 = dst1 + num * (blob->width() + 1) * num_channels +
        num_channels;
    for (int channel = 0; channel < num_channels; ++channel) {
      const Dtype* src3 = src2 + channel * blob->height() * blob->width();
      unsigned char* dst3 = dst2 + channel;
      for (int height = 0; height < blob->height(); ++height) {
        const Dtype* src4 = src3 + height * blob->width();
        unsigned char* dst4 = dst3 +
            (height + 1) * (blob->width() + 1) * num_examples * num_channels
            + num_channels * height;
        for (int width = 0; width < blob->width(); ++width) {
          const Dtype* src = src4 + width;
          unsigned char* dst = dst4 + (width + 1) * num_channels;
          *dst = (*src - min_value) / (max_value - min_value) * 255;
        }
      }
    }
  }

  if (image.cols < kMaxWidth || image.cols > 4 * kMaxWidth) {
    cv::Mat resized;
    cv::Size new_size(kMaxWidth,
        std::max(1.0f,
            static_cast<float>(kMaxWidth) * image.rows / image.cols));
    cv::resize(image, resized, new_size, 0, 0, cv::INTER_NEAREST);
    image = resized;
  }
  vector<uchar> buf;
  CHECK(cv::imencode(".png", image, buf));
  message->set_png(string(buf.begin(), buf.end()));
}

template <typename Dtype>
void BasicVisualizer<Dtype>::CreateLogRecord(shared_ptr<Net<Dtype> > net,
    LogRecord* log_record) {
  log_record->set_type(LogRecord::VISUALIZATION);

  VisualizationRecord* visualization_record = log_record->MutableExtension(
      VisualizationRecord::parent);

  const vector<shared_ptr<Layer<Dtype> > >& layers = net->layers();
  const vector<vector<Blob<Dtype>*> >& top_vecs = net->top_vecs();
  CHECK_EQ(layers.size(), top_vecs.size());
  for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
    shared_ptr<Layer<Dtype> > layer = layers.at(layer_index);
    const string& name = layer->layer_param().name();

    const vector<shared_ptr<Blob<Dtype> > >& blobs = layer->blobs();
    for (size_t blob_index = 0; blob_index < blobs.size(); ++blob_index) {
      shared_ptr<Blob<Dtype> > blob = blobs.at(blob_index);

      BlobSnapshot* weight_snapshot =
          visualization_record->add_weight_snapshots();
      ExtractBlob(name, blob.get(), blob_index, weight_snapshot);
    }

    const vector<Blob<Dtype>*>& top_vec = top_vecs.at(layer_index);
    size_t num_top_vecs = top_vec.size();
    for (size_t top_index = 0; top_index < num_top_vecs; ++top_index) {
      const Blob<Dtype>* blob = top_vec.at(top_index);

      BlobSnapshot* activation_snapshot =
          visualization_record->add_activation_snapshots();
      ExtractBlob(name, blob, top_index, activation_snapshot);
    }
  }
}

INSTANTIATE_CLASS(BasicVisualizer);

}  // namespace caffe
