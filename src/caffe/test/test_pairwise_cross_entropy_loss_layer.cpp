#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pairwise_cross_entropy_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// added by Fuchen Long in 8/19/2016
// for the pairwise cross entropy loss layer test

namespace caffe{

template<typename TypeParam>
class PairWiseCrossEntropyLossLayerTest :public MultiDeviceTest<TypeParam>{
protected:
	PairWiseCrossEntropyLossLayerTest()
		:blob_bottom_data_1_(new Blob<Dtype>(12, 10, 1, 1)),
		blob_bottom_data_2_(new Blob<Dtype>(12, 10, 1, 1)),
		blob_bottom_label_(new Blob<Dtype>(12, 1, 1, 1)),
		blob_top_loss_(new Blob<Dtype>()){
		// Fill the value in the two layer
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		// data_1
		filler.Fill(this->blob_bottom_data_1_);
		blob_bottom_vec_.push_back(blob_bottom_data_1_);
		// data_2
		filler.Fill(this->blob_bottom_data_2_);
		blob_bottom_vec_.push_back(blob_bottom_data_2_);
		// the label
		Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
		for (int i = 0; i < 12; ++i)
			label_data[i] = i % 2;
		blob_bottom_vec_.push_back(blob_bottom_label_);
		// top loss
		blob_top_vec_.push_back(blob_top_loss_);
	}

	virtual ~PairWiseCrossEntropyLossLayerTest(){
		delete blob_bottom_data_1_;
		delete blob_bottom_data_2_;
		delete blob_bottom_label_;
		delete blob_top_loss_;
	}

	void TestForward(){
		//Get the loss without a specific object weight
		LayerParameter layer_param;
		//set some hyper parameter
		PairWiseCrossEntropyLossParameter * pairwise_cross_loss_param = layer_param.mutable_pairwise_cross_entropy_loss_param();
		pairwise_cross_loss_param->set_lamda(2.0);
		PairWiseCrossEntropyLossLayer<Dtype> layer_weight_1(layer_param);
		layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype loss_weight_1 =
			layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		//Get the loss again with a different object weight;
		//check that it is scaled appropriately
		const Dtype kLossWeight = 7.7;
		layer_param.add_loss_weight(kLossWeight);
		PairWiseCrossEntropyLossLayer<Dtype> layer_weight_2(layer_param);
		layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype loss_weight_2 =
			layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype kErrorMargin = 1e-5;
		EXPECT_NEAR(loss_weight_1*kLossWeight, loss_weight_2, kErrorMargin);
		//Make sure the loss is non-trivial
		const Dtype kNonTrivialAbsThresh = 1e-1;
		EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
	}

	Blob<Dtype>* blob_bottom_data_1_;
	Blob<Dtype>* blob_bottom_data_2_;
	Blob<Dtype>* blob_bottom_label_;
	Blob<Dtype>* blob_top_loss_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;


};

TYPED_TEST_CASE(PairWiseCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(PairWiseCrossEntropyLossLayerTest, TestForward){
	this->TestForward();
}

TYPED_TEST(PairWiseCrossEntropyLossLayerTest, TestGradient){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	const Dtype kLossWeight = 3.7;
	layer_param.add_loss_weight(kLossWeight);
	PairWiseCrossEntropyLossParameter * pairwise_cross_loss_param = layer_param.mutable_pairwise_cross_entropy_loss_param();
	pairwise_cross_loss_param->set_lamda(2.0);
	PairWiseCrossEntropyLossLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-2, 1e-4, 1701);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_,0);
}

}// namespace caffe