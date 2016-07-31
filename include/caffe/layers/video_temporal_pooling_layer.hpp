#ifndef CAFFE_VIDEO_TEMPORAL_POOLING_LAYER_HPP_
#define CAFFE_VIDEO_TEMPORAL_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
/**
 * @ brief to temporal pooling the fc layer
 *   and the input feature dim is all_frame_num * channel
 *   the output dim is video_num * channel * 1 *frame_num
 *   Added by Fuchen Long in 8/1/2016
 */

template <typename Dtype>
class VideoTemporalPoolingLayer : public Layer<Dtype>{
public:
	explicit VideoTemporalPoolingLayer(const LayerParameter& param)
		: Layer<Dtype>(param){}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "VideoTemporalPooling"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	int frame_num_;
};

} // namespace caffe


#endif