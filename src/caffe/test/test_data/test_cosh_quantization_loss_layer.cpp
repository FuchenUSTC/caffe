#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cosh_quantization_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// Added by Fuchen Long in 8/9/2016
// to test the cosh_quantization_loss

namespace caffe{

template <typename TypeParam>
class CoshQuantizationLossLayerTest : public MultiDeviceTest<TypeParam>{

protected:
	CoshQuantizationLossLayerTest()
		:blob_bottom_data_(new Blob<Dtype>(12, 24, 1, 1)),
		blob_top_loss_(new Blob<Dtype>()){
		Dtype* bottom_code = blob_bottom_data_->mutable_cpu_data();
		std::srand(time(NULL));
		for (int i = 0; i < 12 * 24; ++i) {
			bottom_code[i] = (std::rand() % 100000)*0.00001;
		}
		blob_bottom_vec_.push_back(blob_bottom_data_);
		blob_top_vec_.push_back(blob_top_loss_);
	}

	virtual ~CoshQuantizationLossLayerTest(){
		delete blob_bottom_data_;
		delete blob_top_loss_;
	}

	void TestForward(){
		LayerParameter layer_param;
		CoshQuantizationLossParameter* cosh_param = layer_param.mutable_cosh_quantization_loss_param();
		cosh_param->set_dim(24);

		CoshQuantizationLossLayer<Dtype> layer_weight_1(layer_param);
		layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype loss_weight_1 =
			layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		const Dtype kLossWeight = 7.7;
		layer_param.add_loss_weight(kLossWeight);
		CoshQuantizationLossLayer<Dtype> layer_weight_2(layer_param);
		layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype loss_weight_2 =
			layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype kErrorMargin = 1e-5;
		EXPECT_NEAR(loss_weight_1*kLossWeight, loss_weight_2, kErrorMargin);
		//Make sure the loss is non-trivial
		const Dtype kNonTrivialAbsThresh = 1e-1;
		EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
	}
	Blob<Dtype>* const blob_bottom_data_;
	Blob<Dtype>* const blob_top_loss_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CoshQuantizationLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CoshQuantizationLossLayerTest, TestForward){
	this->TestForward();
}

TYPED_TEST(CoshQuantizationLossLayerTest, TestGradient){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	const Dtype kLossWeight = 3.7;
	layer_param.add_loss_weight(kLossWeight);
	CoshQuantizationLossParameter* cosh_param = layer_param.mutable_cosh_quantization_loss_param();
	cosh_param->set_dim(24);

	CoshQuantizationLossLayer<Dtype> layer(layer_param);
	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}

}// namespace caffe
