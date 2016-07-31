// This program converts a set of feature to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//    convert_triplet_feature_data.exe [FLAGS] FEATURE_LIST_BIN TRIPLET_LIST OUTPUT_DB
//      DIM FEATURE_NUM TRIPLET_NUM
//
//   ....


// added by Fuchen Long in 7.31.2016
// this is only for reading bin file feature list 

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "boost/scoped_ptr.hpp"
#include "stdint.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

using namespace caffe;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
	"The backend {lmdb,leveldb} for storing the result");



void convert_dataset(const char* feature_list_filename, const char*maping_list_filename,
	const char* db_filename, int feature_dim, int numF, int numTriplet) {

	// Open files
	std::ifstream feature_list_in1(feature_list_filename, std::ios::in); // The feature list
	std::ifstream maping_list_in2(maping_list_filename, std::ios::in);// The maping filename


	CHECK(feature_list_in1) << "Unable to open file " << feature_list_filename; // added by fuchen long
	CHECK(maping_list_in2) << "Unable to open file " << maping_list_filename;
	int  dim = feature_dim;
	int numberFeature = numF;
	int numberTriplet = numTriplet;
	float feature;


	//get the feature
	float **TotalFeature = new float *[numberFeature];
	for (int i = 0; i < numberFeature; i++)
	{
		TotalFeature[i] = new float[dim];
	}

	for (int i = 0; i < numberFeature; i++)
	{
		for (int k = 0; k < dim; k++)
			feature_list_in1 >> TotalFeature[i][k];
		if (i % 1000 == 0)
			LOG(INFO) << "have get the " << i << " features \n";
	}
	LOG(INFO) << "have get the feature\n";
	//get the maping
	int  **TripletMaping = new int *[numberTriplet];
	for (int i = 0; i < numberTriplet; i++)
	{
		TripletMaping[i] = new int[3];
	}
	for (int i = 0; i < numberTriplet; i++)
	{
		for (int k = 0; k < 3; k++)
			maping_list_in2 >> TripletMaping[i][k];
		if (i % 10000 == 0)
			LOG(INFO) << "have get the " << i << " mapings \n";
	}

	// just for the testing
	std::cout << "The feature check " <<
		TotalFeature[55][14] << " " << TotalFeature[90][13] << " \n";
	std::cout << "The maping check " <<
		TripletMaping[15][1] << " " << TripletMaping[123][2] << " \n";
	feature_list_in1.close();
	maping_list_in2.close();



	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(db_filename, db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());


	const int kMaxKeyLength = 256;
	char key[kMaxKeyLength];
	std::string value;
	int index;
	int count = 0;

	for (int triplet = 0; triplet < numberTriplet; ++triplet) {
		caffe::Datum datum;
		datum.set_channels(3 * feature_dim);  // one channel for each image in the pair
		datum.set_height(1);
		datum.set_width(1);


		index = TripletMaping[triplet][0] - 1;
		for (int i = 0; i<dim; i++)
		{
			datum.add_float_data(TotalFeature[index][i]);
		}

		index = TripletMaping[triplet][1] - 1;
		for (int i = 0; i < dim; i++)
		{
			datum.add_float_data(TotalFeature[index][i]);
		}

		index = TripletMaping[triplet][2] - 1;
		for (int i = 0; i<dim; i++)
		{
			datum.add_float_data(TotalFeature[index][i]);
		}
		datum.set_label(1);
		// sequential
		int length = _snprintf(key, kMaxKeyLength, "%08d", triplet);

		// Put in db
		datum.SerializeToString(&value);
		txn->Put(std::string(key, length), value);
		if (++count % 1000 == 0){
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(ERROR) << "triplet:" << count << "\n";
		}
	}

	// write the last batch
	if (count % 1000 != 0){
		txn->Commit();
		LOG(ERROR) << "Success processing " << count << " files.";
	}

	//delelt the db and others

	for (int i = 0; i < numberFeature; i++)
	{
		delete[] TotalFeature[i];
	}
	delete[] TotalFeature;

	for (int i = 0; i < numberTriplet; i++)
	{
		delete[] TripletMaping[i];
	}
	delete[] TripletMaping;

}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of features to the tripelt leveldb/lmdb\n"
		"for the hashing/feature learning.\n"
		"Usage:\n"
		"    convert_triplet_feature_data.exe FEATURE_LIST TRIPLET_LIST OUTPUT_DB\n"
		"DIM FEATURE_NUM TRIPLET_NUM\n"
		"\n"
		);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 7){
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_triplet_feature_data");
		return 1;
	}

	int feature_dim = atoi(argv[4]);
	int number_F = atoi(argv[5]);
	int number_T = atoi(argv[6]);
	convert_dataset(argv[1], argv[2], argv[3], feature_dim, number_F, number_T);
}