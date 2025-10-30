
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vsag/vsag.h>

#include <iostream>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);

    /******************* Create IVF Index *****************/
    std::string ivf_build_params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "buckets_count": 50,
            "base_quantization_type": "fp32",
            "partition_strategy_type": "ivf",
            "ivf_train_type": "kmeans"
            "ivf_train_sample_count": 32767
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("ivf", ivf_build_params).value();

    /******************* Build IVF Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index IVF contains: " << index->GetNumElements() << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    /******************* Prepare Query Dataset *****************/
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    /******************* KnnSearch For IVF Index *****************/
    auto ivf_search_parameters = R"(
    {
        "ivf": {
            "scan_buckets_count": 10
        }
    })";
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, ivf_search_parameters).value();

    /******************* Print Search Result *****************/
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    return 0;
}
