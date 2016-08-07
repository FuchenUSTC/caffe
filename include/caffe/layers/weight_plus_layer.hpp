#ifndef CAFFE_WEIGHT_PLUS_HPP_
#define CAFFE_WEIGHT_PLUS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{

/**
 * @brief Weight plus layer 
 * for the hashing learning, and for bit scaleble hashing
 * Added by Fuchen Long in 8/7/2016
*/

template <typename Dtype>
class WeightPlusLayer :public NeuronLayer<Dtype>{
public: 
	explicit WeightPlusLayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "WeightPlus"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int batch_;
	int dim_;
	Blob<Dtype> weight_pow_;
	Blob<Dtype> weight_two_;
	Blob<Dtype> data_meta_;
};

}// namespace caffe

#endif